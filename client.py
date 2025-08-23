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

def safe_get_hf_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        print(f"❌ Failed to load tokenizer for {model_name}: {e}")
        raise


class ClientLLMModel(nn.Module):
    def __init__(self, model_name, num_prefix=5):
        super(ClientLLMModel, self).__init__()
        print(f"Loading client model: {model_name}")
        
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32, device_map=None
            )
            self.num_prefix = num_prefix
            print(f"✅ Client base model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load client model: {e}")
            raise
        
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        print(f"All client parameters frozen for privacy")
    
    def forward(self, server_output, labels=None):
        try:
            inputs_embeds = server_output['inputs_embeds']
            attention_mask = server_output['attention_mask']
            
            # CRITICAL FIX: Proper label alignment
            if labels is not None:
                batch_size, seq_len = labels.shape
                
                # Add prefix labels (-100 for positions that shouldn't be predicted)
                prefix_labels = torch.full(
                    (batch_size, self.num_prefix), -100,
                    device=labels.device, dtype=labels.dtype
                )
                
                # Concatenate prefix labels with original labels
                full_labels = torch.cat([prefix_labels, labels], dim=1)
                
                # CRITICAL: Ensure input and label lengths match exactly
                input_len = inputs_embeds.shape[1]
                label_len = full_labels.shape[1]
                
                if input_len != label_len:
                    print(f"Adjusting label length: input={input_len}, labels={label_len}")
                    if label_len > input_len:
                        # Truncate labels if too long
                        full_labels = full_labels[:, :input_len]
                    else:
                        # Pad labels if too short
                        pad_length = input_len - label_len
                        padding = torch.full(
                            (batch_size, pad_length), -100,
                            device=labels.device, dtype=labels.dtype
                        )
                        full_labels = torch.cat([full_labels, padding], dim=1)
                
                labels = full_labels
            
            outputs = self.base_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
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


def handle_training_requests_sgd(client_model, optimizer, client, device, args):
    current_labels = None
    current_batch = None  # Store current batch
    batch_count = 0
    recent_losses = []
    recent_accuracies = []
    
    print("Starting SGD training request handler...")
    
    # Get tokenizer for accuracy calculation
    from transformers import AutoTokenizer
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
                print(f"\rProcessing batch {batch_count}...", end='', flush=True)
                
                server_output = {k: v.to(device) if torch.is_tensor(v) else v 
                               for k, v in server_data['data'].items()}
                
                # CRITICAL: Ensure gradients are enabled for server embeddings
                if 'inputs_embeds' in server_output:
                    server_output['inputs_embeds'].requires_grad_(True)
                
                if current_labels is not None:
                    labels = current_labels
                else:
                    print("❌ Error: No labels available for forward pass")
                    client.send_data({'error': 'no_labels'})
                    continue
                
                client_model.train()
                optimizer.zero_grad()
                
                # Forward pass
                outputs = client_model(server_output, labels)
                loss = outputs.loss
                
                # Calculate accuracy with SQUAD-aware method
                accuracy = calculate_accuracy(outputs, labels, current_batch, tokenizer)
                
                recent_losses.append(loss.item())
                recent_accuracies.append(accuracy)
                
                # Keep only last 10 for moving average
                if len(recent_losses) > 10:
                    recent_losses.pop(0)
                    recent_accuracies.pop(0)
                
                avg_recent_loss = sum(recent_losses) / len(recent_losses)
                avg_recent_acc = sum(recent_accuracies) / len(recent_accuracies)
                
                # Backward pass
                loss.backward()
                
                # Handle gradients (same as before)
                if server_output['inputs_embeds'].grad is not None:
                    full_grad = server_output['inputs_embeds'].grad
                    prefix_grad = full_grad[:, :client_model.num_prefix, :]
                    server_grad_flat = prefix_grad.sum(dim=0).flatten()
                    
                    if batch_count < 3:
                        print(f"\n  Client Debug Batch {batch_count}:")
                        print(f"   - Full grad shape: {full_grad.shape}")
                        print(f"   - Prefix grad shape: {prefix_grad.shape}")
                        print(f"   - Server grad flat shape: {server_grad_flat.shape}")
                        print(f"   - Grad norm: {server_grad_flat.norm().item():.6f}")
                else:
                    embedding_dim = server_output['inputs_embeds'].shape[-1]
                    server_grad_flat = torch.zeros(
                        client_model.num_prefix * embedding_dim, device=device
                    )
                    print(f"\n⚠️ No gradients received for batch {batch_count}")
                
                print(f"\rBatch {batch_count}: Loss {loss.item():.4f}, Acc {accuracy:.6f} "
                      f"(Avg: L={avg_recent_loss:.4f}, A={avg_recent_acc:.6f}) - Sending gradients...", 
                      end='', flush=True)
                
                # Send response to server
                client.send_data({
                    'loss': loss.detach().cpu(),
                    'server_grad': server_grad_flat.detach().cpu()
                })
                
                # Update client model
                optimizer.step()
                batch_count += 1
                
                if batch_count % 20 == 0:
                    print(f"\n  Client Progress: {batch_count} batches processed")
                    print(f"    Recent avg - Loss: {avg_recent_loss:.4f}, Acc: {avg_recent_acc:.6f}")
                
            else:
                # Receiving training data
                if batch_count == 0:
                    print(f"Client receiving SQUAD training data...")
                
                input_ids = server_data['input_ids'].to(device)
                attention_mask = server_data['attention_mask'].to(device)
                labels = server_data['labels'].to(device)
                iteration = server_data['iteration']
                
                current_labels = labels
                # Store the current batch for SQUAD processing
                current_batch = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                }
                
                # Add formatted_text if available (for SQUAD)
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
            print(f"❌ ERROR IN CLIENT SQUAD TRAINING: {e}")
            import traceback
            traceback.print_exc()
            break

def parse_args():
    parser = argparse.ArgumentParser(description='Split Learning LLM Client')
    parser.add_argument('--model_name', type=str, default='facebook/opt-125m', help='Pretrained model name')
    parser.add_argument('--num_prefix', type=int, default=5, help='Number of prefix tokens')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    # CRITICAL FIX: Align learning rates
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (match server)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--train_batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=4, help='Test batch size')
    # Training steps
    parser.add_argument('--max_steps', type=int, default=4000, help='Maximum training steps')  # STEPS=4000
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (fallback)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("IMPROVED SPLIT LEARNING CLIENT")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = safe_get_hf_tokenizer(args.model_name)
    print("Tokenizer loaded successfully")
    
    print(f"Creating client model...")
    client_model = ClientLLMModel(args.model_name, args.num_prefix).to(device)
    
    print(f"Model loaded: {args.model_name}")
    print(f"Client trainable parameters: {sum(p.numel() for p in client_model.parameters() if p.requires_grad)}")
    
    # Setup optimizer (even if no trainable params, for consistency)
    client_params = [p for p in client_model.parameters() if p.requires_grad]
    if len(client_params) > 0:
        optimizer = optim.SGD(client_params, lr=args.lr, momentum=args.momentum)
        print(f"Optimizer created for {len(client_params)} trainable parameters")
    else:
        dummy_param = nn.Parameter(torch.tensor(0.0, device=device, requires_grad=True))
        optimizer = optim.SGD([dummy_param], lr=args.lr, momentum=args.momentum)
        print("No trainable parameters found, using dummy optimizer")
    
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
            'lr': args.lr  # Send learning rate for verification
        }
        client.send_data(client_config)
        print(f"Sent client configuration")

        print("=" * 60)
        print("STARTING TRAINING REQUEST HANDLER...")
        print("=" * 60)
        
        handle_training_requests_sgd(client_model, optimizer, client, device, args)
        
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