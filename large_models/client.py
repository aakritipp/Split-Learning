import socket
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import argparse
import traceback

# Import enhanced metrics for proper F1 scoring
from metrics import calculate_client_answer_accuracy

def safe_get_hf_tokenizer(model_name):
    """Keep your existing tokenizer function exactly as is"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        print(f"Failed to load tokenizer for {model_name}: {e}")
        raise


# KEEP your existing ClientLLMModel class - only add minor enhancements
class ClientLLMModel(nn.Module):
    """Enhanced version that maintains full backward compatibility"""
    def __init__(self, model_name, tuning_mode="prefix", num_prefix=5):
        super(ClientLLMModel, self).__init__()
        print(f"Loading client model: {model_name} ({tuning_mode} mode)")
        
        self.tuning_mode = tuning_mode
        self.num_prefix = num_prefix
        
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32, device_map=None
            )
            print(f"Client base model loaded successfully")
        except Exception as e:
            print(f"Failed to load client model: {e}")
            raise
        
        # Freeze all parameters for privacy (keep exactly as is)
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        print(f"All client parameters frozen for privacy")
        print(f"Tuning mode: {tuning_mode}")

    def forward(self, server_output, labels=None):
        try:
            inputs_embeds = server_output['inputs_embeds']
            attention_mask = server_output['attention_mask']
            
            # KEEP your existing label handling logic - it works for SQuAD
            if labels is not None:
                if self.tuning_mode == "prefix":
                    # Your working prefix mode logic
                    batch_size, original_seq_len = labels.shape
                    input_seq_len = inputs_embeds.shape[1]
                    
                    # Create labels that match the input sequence length
                    # Prefix tokens get -100 (ignored in loss)
                    prefix_labels = torch.full(
                        (batch_size, self.num_prefix), -100,
                        device=labels.device, dtype=labels.dtype
                    )
                    
                    # Concatenate prefix labels with original labels
                    adjusted_labels = torch.cat([prefix_labels, labels], dim=1)
                    
                    # Ensure exact length match
                    if adjusted_labels.shape[1] != input_seq_len:
                        if adjusted_labels.shape[1] > input_seq_len:
                            adjusted_labels = adjusted_labels[:, :input_seq_len]
                        else:
                            pad_length = input_seq_len - adjusted_labels.shape[1]
                            padding = torch.full(
                                (batch_size, pad_length), -100,
                                device=labels.device, dtype=labels.dtype
                            )
                            adjusted_labels = torch.cat([adjusted_labels, padding], dim=1)
                    
                    labels = adjusted_labels
                
                elif self.tuning_mode == "lora":
                    # Your working LoRA mode logic
                    input_len = inputs_embeds.shape[1]
                    label_len = labels.shape[1]
                    
                    if input_len != label_len:
                        batch_size = labels.shape[0]
                        if label_len > input_len:
                            labels = labels[:, :input_len]
                        else:
                            pad_length = input_len - label_len
                            padding = torch.full(
                                (batch_size, pad_length), -100,
                                device=labels.device, dtype=labels.dtype
                            )
                            labels = torch.cat([labels, padding], dim=1)
            
            # Forward pass with properly aligned inputs and labels
            outputs = self.base_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False
            )
            
            return outputs
            
        except Exception as e:
            print(f"Client forward pass failed: {e}")
            print(f"   Input embeds shape: {inputs_embeds.shape if 'inputs_embeds' in locals() else 'N/A'}")
            print(f"   Attention mask shape: {attention_mask.shape if 'attention_mask' in locals() else 'N/A'}")
            print(f"   Labels shape: {labels.shape if 'labels' in locals() and labels is not None else 'N/A'}")
            traceback.print_exc()
            raise


# KEEP your existing Client class exactly as is
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
            print(f"Failed to send data: {e}")
            raise
    
    def receive_data(self):
        try:
            length = int.from_bytes(self.socket.recv(4), 'big')
            data = b''
            while len(data) < length:
                data += self.socket.recv(length - len(data))
            return pickle.loads(data)
        except Exception as e:
            print(f"Failed to receive data: {e}")
            raise
    
    def close(self):
        self.socket.close()


def calculate_simple_accuracy(outputs, labels, batch=None, tokenizer=None):
    """
    Enhanced but simple accuracy calculation that works across all tasks
    Maintains compatibility with your existing SQuAD accuracy calculation
    """
    try:
        if outputs.loss is None:
            return 0.0
            
        logits = outputs.logits
        
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
        
        # Simple accuracy calculation
        mask = (shift_labels != -100)
        if mask.sum() > 0:
            correct = (predictions == shift_labels) & mask
            accuracy = correct.sum().float() / mask.sum().float()
            return accuracy.item()
        else:
            return 0.0
        
    except Exception as e:
        print(f"Client accuracy calculation failed: {e}")
        return 0.0


# ENHANCED: Updated training handler that maintains full compatibility
def handle_training_requests_sgd(client_model, optimizer, client, device, args):
    """
    Enhanced training handler that works with all tasks while maintaining SQuAD compatibility
    """
    current_labels = None
    current_batch = {}
    batch_count = 0
    recent_losses = []
    recent_accuracies = []
    tuning_mode = args.tuning_mode  # Get from args instead of default
    
    print(f"Starting SGD training request handler for {args.task.upper()}...")
    
    # Get tokenizer for accuracy calculation
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except:
        tokenizer = None
        print("Could not load tokenizer for client-side accuracy calculation")
    
    while True:
        try:
            server_data = client.receive_data()
            
            # Debug info for first few batches
            if batch_count < 3:
                print(f"\nDEBUG: Message keys = {list(server_data.keys())}")
            
            # Check message type
            if server_data.get('type') == 'training_complete':
                print(f"TRAINING COMPLETED - CLIENT SHUTTING DOWN ({args.task.upper()})")
                break
                
            elif server_data.get('type') == 'forward':
                # Server sends tensors for forward pass
                print(f"Processing {args.task} forward pass for batch {batch_count}...")
                
                # Extract server output
                server_output = {}
                if 'inputs_embeds' in server_data:
                    server_output['inputs_embeds'] = server_data['inputs_embeds'].to(device)
                    server_output['inputs_embeds'].requires_grad_(True)
                if 'attention_mask' in server_data:
                    server_output['attention_mask'] = server_data['attention_mask'].to(device)
                
                # Get labels from the message
                if 'labels' in server_data:
                    labels = server_data['labels'].to(device)
                elif current_labels is not None:
                    labels = current_labels
                else:
                    print("Error: No labels available for forward pass")
                    client.send_data({'error': 'no_labels'})
                    continue
                
                # Update tuning mode from server data if provided
                if 'tuning_mode' in server_data:
                    tuning_mode = server_data['tuning_mode']
                    client_model.tuning_mode = tuning_mode
                
                # Update current batch safely
                if 'labels' in server_data:
                    current_labels = labels
                    current_batch = {
                        'input_ids': server_data.get('input_ids', current_batch.get('input_ids')),
                        'attention_mask': server_data['attention_mask'],
                        'labels': labels
                    }
                    
                    if 'formatted_text' in server_data:
                        current_batch['formatted_text'] = server_data['formatted_text']
                    if 'original_example' in server_data:
                        current_batch['original_example'] = server_data['original_example']
                
                client_model.train()
                optimizer.zero_grad()
                
                # Forward pass
                outputs = client_model(server_output, labels)
                loss = outputs.loss
                
                # Extract predictions for accuracy calculation
                logits = outputs.logits
                
                # Ensure logits and labels have compatible shapes
                if logits.shape[1] != labels.shape[1]:
                    min_len = min(logits.shape[1], labels.shape[1])
                    logits = logits[:, :min_len, :]
                    labels_aligned = labels[:, :min_len]
                else:
                    labels_aligned = labels
                
                # For language modeling: predict next token
                if logits.shape[1] > 1:
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels_aligned[:, 1:].contiguous()
                else:
                    shift_logits = logits
                    shift_labels = labels_aligned
                
                # Get predictions
                predictions = torch.argmax(shift_logits, dim=-1)
                
                # Calculate accuracy using enhanced metrics (includes F1 considerations)
                accuracy = calculate_client_answer_accuracy(
                    predictions, shift_labels, current_batch, tokenizer
                )
                
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
                
                # Handle gradients based on tuning mode
                if server_output['inputs_embeds'].grad is not None:
                    full_grad = server_output['inputs_embeds'].grad
                    
                    if tuning_mode == "prefix":
                        # For prefix mode: extract prefix gradients
                        prefix_grad = full_grad[:, :client_model.num_prefix, :]
                        server_grad_flat = prefix_grad.sum(dim=0).flatten()
                    else:  # LoRA mode
                        # For LoRA mode: use full gradients
                        server_grad_flat = full_grad.sum(dim=(0, 1))
                    
                    if batch_count < 3:
                        print(f"Client Debug Batch {batch_count} ({tuning_mode} mode, {args.task}):")
                        print(f"   - Full grad shape: {full_grad.shape}")
                        if tuning_mode == "prefix":
                            print(f"   - Prefix grad shape: {prefix_grad.shape}")
                        print(f"   - Server grad flat shape: {server_grad_flat.shape}")
                        print(f"   - Grad norm: {server_grad_flat.norm().item():.6f}")
                else:
                    # No gradients case
                    if tuning_mode == "prefix":
                        embedding_dim = server_output['inputs_embeds'].shape[-1]
                        server_grad_flat = torch.zeros(
                            client_model.num_prefix * embedding_dim, device=device
                        )
                    else:  # LoRA mode
                        embedding_dim = server_output['inputs_embeds'].shape[-1]
                        server_grad_flat = torch.zeros(embedding_dim, device=device)
                    
                    print(f"No gradients received for batch {batch_count}")
                
                print(f"{args.task} Batch {batch_count}: Loss {loss.item():.4f}, Acc {accuracy:.6f} "
                      f"(Avg: L={avg_recent_loss:.4f}, A={avg_recent_acc:.6f})")
                
                # Send response to server
                client.send_data({
                    'loss': loss.detach().cpu(),
                    'server_grad': server_grad_flat.detach().cpu()
                })
                
                # Update client model (no-op since all params frozen)
                optimizer.step()
                batch_count += 1
                
                if batch_count % 20 == 0:
                    print(f"Client Progress ({args.task}): {batch_count} batches processed")
                    print(f"    Recent avg - Loss: {avg_recent_loss:.4f}, Acc: {avg_recent_acc:.6f}")
                    
            else:
                # This is initial training data (first batch)
                if batch_count == 0:
                    print(f"Client receiving {args.task.upper()} training data...")
                    # Extract tuning mode from server data if available
                    if 'tuning_mode' in server_data:
                        tuning_mode = server_data['tuning_mode']
                        client_model.tuning_mode = tuning_mode
                        print(f"Client tuning mode set to: {tuning_mode}")
                
                # Extract training data
                input_ids = server_data['input_ids'].to(device)
                attention_mask = server_data['attention_mask'].to(device)
                labels = server_data['labels'].to(device)
                iteration = server_data.get('iteration', batch_count)
                
                current_labels = labels
                # Store the current batch
                current_batch = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                }
                
                # Add task-specific data if available
                if 'formatted_text' in server_data:
                    current_batch['formatted_text'] = server_data['formatted_text']
                if 'original_example' in server_data:
                    current_batch['original_example'] = server_data['original_example']
                
                if iteration == 0:
                    print(f"Starting {args.task.upper()} training - Batch size: {input_ids.shape[0]}, "
                          f"Sequence length: {input_ids.shape[1]}")
                    print(f"Tuning mode: {tuning_mode}")
                    if 'formatted_text' in server_data and len(server_data['formatted_text']) > 0:
                        print(f"Sample {args.task} text: {server_data['formatted_text'][0][:150]}...")
                        
        except Exception as e:
            print(f"ERROR IN CLIENT {args.task.upper()} TRAINING: {e}")
            import traceback
            traceback.print_exc()
            break


def parse_args():
    """Enhanced argument parser with task support"""
    parser = argparse.ArgumentParser(description='Enhanced Split Learning LLM Client with Multi-Task support')
    parser.add_argument('--model_name', type=str, default='facebook/opt-125m', help='Pretrained model name')
    parser.add_argument('--tuning_mode', type=str, default='prefix', choices=['prefix', 'lora'], help='Fine-tuning method')
    parser.add_argument('--num_prefix', type=int, default=5, help='Number of prefix tokens')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (match server)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--train_batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=4, help='Test batch size')
    parser.add_argument('--max_steps', type=int, default=4000, help='Maximum training steps')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (fallback)')
    
    # NEW: Task selection (should match server)
    parser.add_argument('--task', type=str, default='squad', 
                       choices=['squad', 'drop', 'sst2'], 
                       help='Dataset task to use (must match server)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 80)
    print(f"ENHANCED SPLIT LEARNING CLIENT ({args.task.upper()})")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = safe_get_hf_tokenizer(args.model_name)
    print("Tokenizer loaded successfully")
    
    print(f"Creating client model for {args.task.upper()}...")
    client_model = ClientLLMModel(args.model_name, args.tuning_mode, args.num_prefix).to(device)

    print(f"Model loaded: {args.model_name}")
    print(f"Task: {args.task.upper()}")
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
        print("=" * 80)
        print(f"CLIENT STARTING - ATTEMPTING TO CONNECT TO SERVER ({args.task.upper()})")
        print("=" * 80)
        print("Trying to connect to server at localhost:12345...")
        
        client = Client('localhost', 12345)
        
        print("=" * 80)
        print(f"CLIENT SUCCESSFULLY CONNECTED TO SERVER! ({args.task.upper()})")
        print("=" * 80)

        # Send client configuration including task info
        client_config = {
            'model_name': args.model_name,
            'tuning_mode': args.tuning_mode,
            'num_prefix': args.num_prefix,
            'lr': args.lr,
            'task': args.task  # Include task information
        }
        client.send_data(client_config)
        print(f"Sent client configuration for {args.task.upper()}")

        print("=" * 80)
        print(f"STARTING {args.task.upper()} TRAINING REQUEST HANDLER...")
        print("=" * 80)
        
        handle_training_requests_sgd(client_model, optimizer, client, device, args)
        
    except ConnectionRefusedError:
        print("=" * 80)
        print("CONNECTION FAILED!")
        print("=" * 80)
        print("Could not connect to server at localhost:12345")
        print("Make sure the server is running first:")
        print(f"   python server.py --model_name {args.model_name} --tuning_mode {args.tuning_mode} --task {args.task}")
        print("=" * 80)
    except Exception as e:
        print(f"CLIENT ERROR: {e}")
        traceback.print_exc()
    finally:
        try:
            client.close()
        except:
            pass
        print(f"CLIENT SHUTDOWN COMPLETE ({args.task.upper()})")