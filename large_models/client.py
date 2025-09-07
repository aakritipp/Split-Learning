import socket
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import argparse
import traceback
from metrics import calculate_client_answer_accuracy
from SGDGradientEst import StochasticGradientApproximator

# Global KV model holder (set in main)
kv_model = None

try:
    from transformers.cache_utils import DynamicCache, StaticCache
except Exception:
    DynamicCache = None
    StaticCache = None


def _normalize_batch_from_server(payload, device, tokenizer):
    ids = payload['input_ids'].to(device)
    am  = payload['attention_mask'].to(device)

    if 'labels' in payload and torch.is_tensor(payload['labels']):
        labels = payload['labels'].to(device).long()
    else:
        # Build labels from input_ids; ignore pads with -100 (HF causal LM shifts internally)
        labels = ids.clone()
        labels[am == 0] = -100
        pad_id = getattr(tokenizer, 'pad_token_id', None)
        if pad_id is not None:
            labels[ids == pad_id] = -100
        labels = labels.long()

    kv_state = payload.get('server_kv_state', None)
    return ids, am, labels, kv_state


def safe_get_hf_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        print(f"❌ Failed to load tokenizer for {model_name}: {e}")
        raise


class PrefixEncoder(nn.Module):
    """Prefix encoder for client - same as server"""
    def __init__(self, hidden_size, num_prefix=5):
        super(PrefixEncoder, self).__init__()
        self.num_prefix = num_prefix
        self.hidden_size = hidden_size
        
        # Initialize prefix embeddings
        self.prefix_embeddings = nn.Parameter(
            torch.randn(num_prefix, hidden_size) * (hidden_size ** -0.5)
        )
        
        # Better initialization
        with torch.no_grad():
            nn.init.normal_(self.prefix_embeddings, mean=0.0, std=0.02)
        
        print(f"  Client PrefixEncoder: {num_prefix} tokens x {hidden_size} dims = {num_prefix * hidden_size} parameters")
        print(f"  Client prefix embedding std: {self.prefix_embeddings.std().item():.6f}")
    
    def forward(self, batch_size):
        """Expand prefix embeddings for the given batch size"""
        return self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)



import types
from prefix_kv import PrefixKV, merge_past_key_values, flatten_grad_state

class ClientKVModel(nn.Module):
    """
    Client-side model that supports per-layer KV-prefix (only prefixes are trainable).
    It holds two PrefixKV modules:
      - server_kv_mirror: non-owned copy used to compute gradients for server (when server uses SGD)
      - client_kv: trainable prefixes for the client's layers
    We still run the full model on the client, but inject KV prefixes into all layers.
    """
    def __init__(self, model_name, total_layers, cut_layer, num_prefix=10):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map=None)
        for p in self.base_model.parameters():
            p.requires_grad = False

        self.total_layers = total_layers
        self.cut_layer = cut_layer
        self.hidden_size = self.base_model.config.hidden_size

        # server prefixes live on layers [0..cut-1]; client prefixes live on [cut..L-1]
        self.server_kv_mirror = PrefixKV(self.base_model.config, list(range(0, cut_layer)), num_prefix=num_prefix, device=self.base_model.device)
        self.client_kv = PrefixKV(self.base_model.config, list(range(cut_layer, total_layers)), num_prefix=num_prefix, device=self.base_model.device)

    def load_server_state(self, state_dict: dict):
        # state_dict expected to contain keys "k" and "v" tensors matching server_kv_mirror
        with torch.no_grad():
            if "k" in state_dict:
                self.server_kv_mirror.k.copy_(state_dict["k"].to(self.server_kv_mirror.k.device, dtype=self.server_kv_mirror.k.dtype))
            if "v" in state_dict:
                self.server_kv_mirror.v.copy_(state_dict["v"].to(self.server_kv_mirror.v.device, dtype=self.server_kv_mirror.v.dtype))

    def forward_full(self, input_ids, attention_mask, labels=None, require_server_grad=False):
        bsz = input_ids.size(0)

        # ---- 1) trim right padding (you already had this, keep it) ----
        if attention_mask is not None:
            valid_len = int(attention_mask.sum(dim=1).max().item())
            valid_len = max(valid_len, 1)
            if valid_len < input_ids.size(1):
                input_ids = input_ids[:, :valid_len]
                attention_mask = attention_mask[:, :valid_len]
                if labels is not None:
                    labels = labels[:, :valid_len]
        seq_len = input_ids.size(1)
        # ---------------------------------------------------------------

        # ---- 2) build KV cache from prefixes (server + client) ----
        server_past = self.server_kv_mirror.get_local_past(bsz)
        client_past = self.client_kv.get_local_past(bsz)
        past_kv = merge_past_key_values(self.total_layers, server_past, client_past)
        self.server_kv_mirror.set_requires_grad(require_server_grad)

        # wrap to HF Cache if available (StaticCache/DynamicCache), else legacy tuple
        legacy = tuple(past_kv)
        cache = legacy
        try:
            if StaticCache is not None and hasattr(StaticCache, "from_legacy_cache"):
                cache = StaticCache.from_legacy_cache(legacy)
            elif DynamicCache is not None and hasattr(DynamicCache, "from_legacy_cache"):
                cache = DynamicCache.from_legacy_cache(legacy)
        except Exception:
            cache = legacy
        # -----------------------------------------------------------

        # ---- 3) compute past length (prefix length) for positions ----
        try:
            past_len = cache.get_seq_length()  # new HF Cache API
        except Exception:
            # derive from first layer K: [B, H, past_len, D]
            first_k = legacy[0][0] if isinstance(legacy, (list, tuple)) else None
            past_len = int(first_k.shape[2]) if isinstance(first_k, torch.Tensor) else 0
        # --------------------------------------------------------------

        # ---- 4) explicit position_ids with correct offset; no attention_mask ----
        position_ids = torch.arange(
            past_len, past_len + seq_len, device=input_ids.device, dtype=torch.long
        ).unsqueeze(0).expand(bsz, -1)

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=None,         # we trimmed pads; avoid HF’s mask-based pos path
            position_ids=position_ids,   # explicit, matches seq_len with past offset
            labels=labels,
            past_key_values=cache,
            use_cache=False,
        )
        return outputs

    def zero_prefix_grads(self):
        for mod in [self.server_kv_mirror, self.client_kv]:
            if hasattr(mod.k, "grad") and mod.k.grad is not None:
                mod.k.grad.zero_()
            if hasattr(mod.v, "grad") and mod.v.grad is not None:
                mod.v.grad.zero_()

class ClientLLMModel(nn.Module):
    def __init__(self, model_name, num_prefix=5):
        super(ClientLLMModel, self).__init__()
        print(f"Loading client model: {model_name}")
        
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32, device_map=None
            )
            self.num_prefix = num_prefix
            
            # Get hidden size from config
            self.hidden_size = self.base_model.config.hidden_size
            
            # Create client's own prefix encoder (trainable)
            
            print(f"✅ Client base model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load client model: {e}")
            raise
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        print(f"Base model parameters frozen for privacy")
    
    def forward(self, server_output, labels=None):
        try:
            # Get server embeddings (already includes server prefix)
            attention_mask = server_output['attention_mask']
            
            batch_size = server_embeds.shape[0]
            
            # Get client prefix embeddings
            
            # Concatenate client prefix with server embeddings
            # Final order: [server_prefix + original_input] + [client_prefix]
            combined_embeds = torch.cat([server_embeds, client_prefix_embeds], dim=1)
            
            # Extend attention mask for client prefix
            client_prefix_mask = torch.ones(
                batch_size, self.num_prefix, 
                device=attention_mask.device, dtype=attention_mask.dtype
            )
            combined_attention_mask = torch.cat([attention_mask, client_prefix_mask], dim=1)
            
            # Handle labels if provided
            if labels is not None:
                batch_size, seq_len = labels.shape
                
                # Add prefix labels for both server and client prefixes
                # Server prefix was already handled, now add client prefix
                total_prefix_len = self.num_prefix  # Only client prefix to add
                prefix_labels = torch.full(
                    (batch_size, total_prefix_len), -100,
                    device=labels.device, dtype=labels.dtype
                )
                
                # Server already added its prefix labels, so we receive labels that include them
                # We just need to add client prefix labels at the end
                full_labels = torch.cat([labels, prefix_labels], dim=1)
                
                # Ensure length consistency
                input_len = combined_embeds.shape[1]
                label_len = full_labels.shape[1]
                
                if input_len != label_len:
                    if label_len > input_len:
                        full_labels = full_labels[:, :input_len]
                    else:
                        pad_length = input_len - label_len
                        padding = torch.full(
                            (batch_size, pad_length), -100,
                            device=labels.device, dtype=labels.dtype
                        )
                        full_labels = torch.cat([full_labels, padding], dim=1)
                
                labels = full_labels
            
            # Forward through base model
            outputs = self.base_model(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention_mask,
                labels=labels,
                use_cache=False
            )
            
            return outputs
            
        except Exception as e:
            print(f"❌ Client forward pass failed: {e}")
            traceback.print_exc()
            raise


class Client:
    def __init__(self, host, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))
        
    def send_data(self, data):
        try:
            serialized = pickle.dumps(data)
            self.socket.sendall(len(serialized).to_bytes(4, 'big'))
            self.socket.sendall(serialized)
        except Exception as e:
            print(f"❌ Failed to send data: {e}")
            raise
    
    def receive_data(self):
        try:
            length = int.from_bytes(self.socket.recv(4), 'big')
            data = b''
            while len(data) < length:
                data += self.socket.recv(length - len(data))
            return pickle.loads(data)
        except Exception as e:
            print(f"❌ Failed to receive data: {e}")
            raise
    
    def close(self):
        self.socket.close()


def calculate_accuracy(outputs, labels, batch=None, tokenizer=None):
    """Calculate SQUAD-specific accuracy from model outputs"""
    try:
        if outputs.loss is None:
            return 0.0
            
        logits = outputs.logits
        
        # Debug for first few calls
        global client_debug_count
        if 'client_debug_count' not in globals():
            client_debug_count = 0
            
        if client_debug_count < 3:
            print(f"\n  Client SQUAD Debug call {client_debug_count}:")
            print(f"   Logits shape: {logits.shape}")
            print(f"   Labels shape: {labels.shape}")
            if batch is not None and 'formatted_text' in batch:
                print(f"   Sample text: {batch['formatted_text'][0][:100]}...")
        
        # Ensure logits and labels have compatible shapes
        if logits.shape[1] != labels.shape[1]:
            min_len = min(logits.shape[1], labels.shape[1])
            logits = logits[:, :min_len, :]
            labels = labels[:, :min_len]
        
        # For language modeling: predict next token
        if logits.shape[1] > 1:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
        else:
            shift_logits = logits
            shift_labels = labels
        
        # Get predictions
        predictions = torch.argmax(shift_logits, dim=-1)
        
        # Use the imported function for SQUAD-specific accuracy calculation
        answer_accuracy = calculate_client_answer_accuracy(
            predictions, shift_labels, batch, tokenizer
        )
        
        if client_debug_count < 3:
            print(f"   Answer accuracy: {answer_accuracy:.6f}")
        
        client_debug_count += 1
        return answer_accuracy
        
    except Exception as e:
        print(f"❌ Client SQUAD accuracy calculation failed: {e}")
        return 0.0


def handle_training_requests_zoo(client_model, optimizer, grad_estimator, client, device, args):
    """Handle training with ZOO - FIXED for mixed optimizer scenarios"""
    current_labels = None
    current_batch = None
    batch_count = 0
    recent_losses = []
    recent_accuracies = []
    
    print("Starting ZOO training request handler...")
    print(f"  ZOO Configuration:")
    print(f"    Perturbation scale (mu): {args.mu}")
    print(f"    Number of perturbations: {args.num_pert}")
    
    # Get tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except:
        tokenizer = None
        print("⚠️ Could not load tokenizer")
    
    while True:
        try:
            server_data = client.receive_data()
            
            if server_data.get('type') == 'training_complete':
                print("TRAINING COMPLETED - CLIENT SHUTTING DOWN")
                break
                
            elif server_data.get('type') == 'forward':
                print(f"\rProcessing batch {batch_count} (Client ZOO)...", end='', flush=True)

                payload = server_data['data']
                ids, am, labels, kv_state = _normalize_batch_from_server(payload, device, tokenizer)
                if kv_state is not None:
                    kv_model.load_server_state(kv_state)

                # If the server is probing the objective for ZOO, do forward-only and return just the loss
                if payload.get('zoo_eval', False):
                    kv_model.eval()
                    with torch.no_grad():
                        out = kv_model.forward_full(ids, am, labels=labels, require_server_grad=False)
                        loss_val = float(out.loss.item())
                    client.send_data({'loss': loss_val})
                    continue

                # Client-side ZOO step (optimize client prefixes only)
                kv_model.train()
                optimizer.zero_grad()

                def client_objective_fn(_x=None, _y=None):
                    # No grads needed for objective evaluation
                    kv_model.eval()
                    with torch.no_grad():
                        out = kv_model.forward_full(ids, am, labels=labels, require_server_grad=False)
                        return out.loss

                # Make sure the estimator knows which params to perturb
                grad_estimator.model_params = list(kv_model.client_kv.parameters())

                # >>> FIX: pass (input_batch, target_labels, objective_fn, random_seed)
                grad_estimator.estimate_gradients(
                    ids, labels, client_objective_fn,
                    random_seed=batch_count * 1000 + args.seed
                )
                optimizer.step()

                # Optional metrics after step
                kv_model.eval()
                with torch.no_grad():
                    metrics_out = kv_model.forward_full(ids, am, labels=labels, require_server_grad=False)
                    loss_val = float(metrics_out.loss.item())

                # If the server is training with SGD (expects server grads), compute & send them.
                server_grad_state = None
                if 'server_kv_state' in payload:
                    kv_model.zero_prefix_grads()
                    out_bp = kv_model.forward_full(ids, am, labels=labels, require_server_grad=True)
                    out_bp.loss.backward()
                    server_grad_state = flatten_grad_state(kv_model.server_kv_mirror)

                resp = {'loss': loss_val}
                if server_grad_state is not None:
                    resp['server_grad_state'] = server_grad_state
                client.send_data(resp)

                batch_count += 1


            else:
                # Receiving training data
                if batch_count == 0:
                    print(f"Client receiving SQUAD training data (Client uses ZOO)...")
                
                input_ids = server_data['input_ids'].to(device)
                attention_mask = server_data['attention_mask'].to(device)
                labels = server_data['labels'].to(device)
                iteration = server_data['iteration']
                
                current_labels = labels
                current_batch = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                }
                
                if 'formatted_text' in server_data:
                    current_batch['formatted_text'] = server_data['formatted_text']
                if 'original_example' in server_data:
                    current_batch['original_example'] = server_data['original_example']
                
                if iteration == 0:
                    print(f"Starting SQUAD training (Client ZOO, Server may use SGD or ZOO)")
                    print(f"  Batch size: {input_ids.shape[0]}, Sequence length: {input_ids.shape[1]}")
                        
        except Exception as e:
            print(f"❌ ERROR IN CLIENT ZOO TRAINING: {e}")
            import traceback
            traceback.print_exc()
            break


def handle_training_requests_sgd(client_model, optimizer, client, device, args):
    """Handle training with standard SGD (backpropagation) - ENHANCED VERSION"""
    current_labels = None
    current_batch = None
    batch_count = 0
    recent_losses = []
    recent_accuracies = []
    
    print("Starting SGD training request handler...")
    
    # Get tokenizer for accuracy calculation
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except:
        tokenizer = None
        print("⚠️ Could not load tokenizer for client-side accuracy calculation")
    
    while True:
        try:
            server_data = client.receive_data()
            
            if server_data.get('type') == 'training_complete':
                print("TRAINING COMPLETED - CLIENT SHUTTING DOWN")
                break
                
            elif server_data.get('type') == 'forward':
                print(f"\rProcessing batch {batch_count} (Client SGD)...", end='', flush=True)

                # payload comes in under the 'data' key
                payload = server_data['data']

                # Normalize inputs (+labels fallback) and load KV prefixes
                ids, am, labels, kv_state = _normalize_batch_from_server(payload, device, tokenizer)
                if kv_state is not None:
                    kv_model.load_server_state(kv_state)

                kv_model.train()
                optimizer.zero_grad()

                # Compute forward on client; require_server_grad=True so we get grads for server KV
                outputs = kv_model.forward_full(ids, am, labels=labels, require_server_grad=True)
                loss = outputs.loss

                # (Optional) Accuracy for logs
                accuracy = calculate_accuracy(outputs, labels, payload, tokenizer)

                # Backprop — this creates grads on kv_model.server_kv_mirror.{k,v}
                loss.backward()

                # Extract KV grads for server (structured)
                server_grad_state = flatten_grad_state(kv_model.server_kv_mirror)

                # Send back loss + server KV grads
                client.send_data({'loss': loss.detach().cpu(), 'server_grad_state': server_grad_state})

                # Step client optimizer (only client prefixes are trainable)
                optimizer.step()

                batch_count += 1

                # Light progress output
                recent_losses.append(loss.item())
                recent_accuracies.append(accuracy)
                if len(recent_losses) > 10:
                    recent_losses.pop(0)
                    recent_accuracies.pop(0)
                avg_recent_loss = sum(recent_losses) / len(recent_losses)
                avg_recent_acc  = sum(recent_accuracies) / len(recent_accuracies)
                print(f"\rBatch {batch_count-1}: Loss {loss.item():.4f}, Acc {accuracy:.6f} "
                    f"(Avg: L={avg_recent_loss:.4f}, A={avg_recent_acc:.6f})", end='', flush=True)

            else:
                # Receiving training data
                if batch_count == 0:
                    print(f"Client receiving SQUAD training data (Client uses SGD)...")
                
                input_ids = server_data['input_ids'].to(device)
                attention_mask = server_data['attention_mask'].to(device)
                labels = server_data['labels'].to(device)
                iteration = server_data['iteration']
                
                current_labels = labels
                current_batch = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                }
                
                if 'formatted_text' in server_data:
                    current_batch['formatted_text'] = server_data['formatted_text']
                if 'original_example' in server_data:
                    current_batch['original_example'] = server_data['original_example']
                
                if iteration == 0:
                    print(f"Starting SQUAD training - Batch size: {input_ids.shape[0]}, "
                          f"Sequence length: {input_ids.shape[1]}")
                    if 'formatted_text' in server_data:
                        print(f"Sample SQUAD text: {server_data['formatted_text'][0][:150]}...")
                        
        except Exception as e:
            print(f"❌ ERROR IN CLIENT SGD TRAINING: {e}")
            traceback.print_exc()
            break


def parse_args():
    parser = argparse.ArgumentParser(description='Split Learning LLM Client')
    parser.add_argument('--model_name', type=str, default='facebook/opt-125m', help='Pretrained model name')
    parser.add_argument('--num_prefix', type=int, default=5, help='Number of prefix tokens')
    parser.add_argument('--cut_layer', type=int, default=6, help='Split index: 0..cut-1 server; cut..L-1 client')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--zoo_lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--train_batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=4, help='Test batch size')
    parser.add_argument('--max_steps', type=int, default=4000, help='Maximum training steps')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (fallback)')
    # ZOO parameters
    parser.add_argument('--mu', type=float, default=1e-1, help='ZOO perturbation scale')
    parser.add_argument('--num_pert', type=int, default=5, help='Number of ZOO perturbations')
    parser.add_argument('--use_zeroth_order_client', action='store_true', help='Use ZOO for client')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("SPLIT LEARNING CLIENT WITH PREFIX TUNING")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = safe_get_hf_tokenizer(args.model_name)
    print("Tokenizer loaded successfully")
    
    print("Creating client KV-prefix model...")
    # KV prefix model (new): always create so handlers can access
    from transformers import AutoConfig
    tmp_cfg = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32, device_map=None).config
    total_layers = tmp_cfg.num_hidden_layers
    from prefix_kv import PrefixKV
    kv_model = ClientKVModel(args.model_name, total_layers, args.cut_layer, num_prefix=args.num_prefix).to(device)
    trainable_params = list(kv_model.client_kv.parameters())

    
    print(f"Model loaded: {args.model_name}")
    
    # Setup optimizer for client's prefix embeddings
    if args.use_zeroth_order_client:
        optimizer = optim.SGD(kv_model.client_kv.parameters(), lr=args.zoo_lr, momentum=0.0)
    else:
        optimizer = optim.SGD(kv_model.client_kv.parameters(), lr=args.lr, momentum=args.momentum)
    print(f"Optimizer created for client prefix embeddings")
    
    # Setup ZOO gradient estimator if needed
    grad_estimator = None
    if args.use_zeroth_order_client:
        print("Setting up ZOO gradient estimator...")
        grad_estimator = StochasticGradientApproximator(
            model_params=trainable_params,
            perturbation_scale=args.mu,
            sample_count=args.num_pert,
            compute_device=device,
            data_type=torch.float32
        )
        print(f"ZOO gradient estimator created")
    
    try:
        print("=" * 60)
        print("CLIENT STARTING - ATTEMPTING TO CONNECT TO SERVER")
        print("=" * 60)
        print("Trying to connect to server at localhost:12345...")
        
        client = Client('localhost', 12345)
        
        print("=" * 60)
        print("CLIENT SUCCESSFULLY CONNECTED TO SERVER!")
        print("=" * 60)

        client_config = {
            'model_name': args.model_name,
            'num_prefix': args.num_prefix,
            'lr': args.lr,
            'use_zeroth_order_client': args.use_zeroth_order_client
        }
        client.send_data(client_config)
        print(f"Sent client configuration")

        print("=" * 60)
        if args.use_zeroth_order_client:
            print("STARTING ZOO TRAINING REQUEST HANDLER...")
            print(f"  ZOO Configuration:")
            print(f"    Perturbation scale (mu): {args.mu}")
            print(f"    Number of perturbations: {args.num_pert}")
        else:
            print("STARTING SGD TRAINING REQUEST HANDLER...")
        print("=" * 60)
        
        # Choose handler based on optimization method
        if args.use_zeroth_order_client:
            handle_training_requests_zoo(
                kv_model, optimizer, grad_estimator, 
                client, device, args
            )
        else:
            handle_training_requests_sgd(
                kv_model, optimizer, client, device, args
            )
        
    except ConnectionRefusedError:
        print("=" * 60)
        print("❌ CONNECTION FAILED!")
        print("=" * 60)
        print("Could not connect to server at localhost:12345")
        print("Make sure the server is running first:")
        print(f"   python server.py --model_name {args.model_name}")
        print("=" * 60)
    except Exception as e:
        print(f"❌ CLIENT ERROR: {e}")
        traceback.print_exc()
    finally:
        try:
            client.close()
        except:
            pass
        print("CLIENT SHUTDOWN COMPLETE")
