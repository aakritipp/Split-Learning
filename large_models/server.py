import socket
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import argparse
import sys
import traceback

# Import your existing modules (don't change these imports)
from SGDGradientEst import StochasticGradientApproximator
from lora import LoRA, LoRALinear, find_module

# Enhanced imports - add these to your working version
from dataset import (
    get_squad_dataloaders,           # Your existing working function
    get_enhanced_dataloaders,        # New enhanced function
    get_hf_tokenizer,               # Your existing function
    SQuADDataset,                   # Your existing class
    MultiTaskDataset                # New multi-task class
)

# Enhanced metrics with proper F1 scoring
from metrics import (
    calculate_enhanced_metrics,      # Main metrics function with F1
    calculate_generation_metrics,    # Generation-based F1/EM
    calculate_client_answer_accuracy # Enhanced client accuracy
)


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


# KEEP your existing squad_collate_fn exactly as is
def squad_collate_fn(batch):
    """Your existing collate function - don't change this!"""
    try:
        input_ids = []
        attention_masks = []
        labels = []
        formatted_texts = []
        original_examples = []
        
        for item in batch:
            input_ids.append(item['input_ids'])
            attention_masks.append(item['attention_mask'])
            labels.append(item['labels'])
            formatted_texts.append(item.get('formatted_text', ""))
            original_examples.append(item.get('original_example', {}))
        
        batch_dict = {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks),
            'labels': torch.stack(labels),
        }
        
        if any(text for text in formatted_texts):
            batch_dict['formatted_text'] = formatted_texts
        if any(ex for ex in original_examples):
            batch_dict['original_example'] = original_examples
        
        return batch_dict
        
    except Exception as e:
        print(f"Collate function error: {e}")
        print(f"Batch info: {len(batch)} items")
        for i, item in enumerate(batch):
            print(f"  Item {i}: keys={list(item.keys())}, shapes={[item[k].shape if torch.is_tensor(item[k]) else type(item[k]) for k in item.keys()]}")
        raise


# ENHANCED: Updated get_dataloaders function
def get_dataloaders(args, tokenizer):
    """Enhanced dataloader creation that maintains SQuAD compatibility"""
    print(f"Creating dataset for task: {args.task}")
    print(f"Train examples: {getattr(args, 'train_examples', 1000)}")
    print(f"Eval examples: {getattr(args, 'eval_examples', 200)}")
    print(f"Batch size: {args.train_batch_size}")
    
    try:
        # Use enhanced function for all tasks (including SQuAD)
        train_loader, eval_loader = get_enhanced_dataloaders(
            task=args.task,
            tokenizer=tokenizer,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            max_length=args.max_length,
            num_train_examples=getattr(args, 'train_examples', 1000),
            num_eval_examples=getattr(args, 'eval_examples', 200)
        )
        
        print(f"{args.task.upper()} dataloaders created successfully")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Eval batches: {len(eval_loader)}")
        
        # Test the dataloader
        print("  Testing dataloader...")
        try:
            test_batch = next(iter(train_loader))
            print(f"Dataloader test passed")
            print(f"   Batch keys: {list(test_batch.keys())}")
            print(f"   Batch shapes: {[f'{k}: {v.shape if torch.is_tensor(v) else len(v)}' for k, v in test_batch.items()]}")
            
            # Show sample text for verification
            if 'formatted_text' in test_batch:
                print(f"   Sample text: {test_batch['formatted_text'][0][:150]}...")
                
        except Exception as test_error:
            print(f"Dataloader test failed: {test_error}")
            raise
        
        return train_loader, eval_loader
        
    except Exception as e:
        print(f"Dataset creation failed: {e}")
        traceback.print_exc()
        raise


# KEEP all your existing model classes exactly as they are
class ServerLLMModel(nn.Module):
    """Keep your existing ServerLLMModel class exactly as is"""
    def __init__(self, model_name, tuning_mode="prefix", num_prefix=5, lora_r=8, lora_alpha=16):
        super(ServerLLMModel, self).__init__()
        self.tuning_mode = tuning_mode
        self.num_prefix = num_prefix
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map=None
        )
        if tuning_mode == "prefix":
            self.prefix_encoder = nn.Embedding(num_prefix, self.base_model.config.hidden_size)
        elif tuning_mode == "lora":
            LoRA(self.base_model, r=lora_r, alpha=lora_alpha, float16=False)
    
    def forward(self, input_ids, attention_mask):
        if self.tuning_mode == "prefix":
            batch_size = input_ids.shape[0]
            prefix_embeds = self.prefix_encoder(
                torch.arange(self.num_prefix, device=input_ids.device).expand(batch_size, -1)
            )
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)
            prefix_mask = torch.ones(batch_size, self.num_prefix, device=attention_mask.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        
        return {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask
        }


class FullLLMModel(nn.Module):
    """Keep your existing FullLLMModel class exactly as is"""
    def __init__(self, model_name, tuning_mode="prefix", num_prefix=5, lora_r=8, lora_alpha=16):
        super(FullLLMModel, self).__init__()
        self.tuning_mode = tuning_mode
        self.num_prefix = num_prefix
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map=None
        )
        if tuning_mode == "prefix":
            self.prefix_encoder = nn.Embedding(num_prefix, self.base_model.config.hidden_size)
        elif tuning_mode == "lora":
            LoRA(self.base_model, r=lora_r, alpha=lora_alpha, float16=False)
    
    def forward(self, input_ids, attention_mask, labels=None):
        if self.tuning_mode == "prefix":
            batch_size = input_ids.shape[0]
            prefix_embeds = self.prefix_encoder(
                torch.arange(self.num_prefix, device=input_ids.device).expand(batch_size, -1)
            )
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)
            prefix_mask = torch.ones(batch_size, self.num_prefix, device=attention_mask.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            if labels is not None:
                prefix_labels = torch.full((batch_size, self.num_prefix), -100, dtype=torch.long, device=labels.device)
                labels = torch.cat([prefix_labels, labels], dim=1)
        else:
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        
        return self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False
        )


# KEEP your existing utility functions
def get_server_trainable_params(server_model):
    """Keep this exactly as is"""
    if server_model.tuning_mode == "prefix":
        return [p for p in server_model.prefix_encoder.parameters() if p.requires_grad]
    else:
        return [p for p in server_model.base_model.parameters() if p.requires_grad]


def sync_models(server_model, full_model):
    """Keep this exactly as is"""
    if server_model.tuning_mode == "prefix":
        full_model.prefix_encoder.load_state_dict(server_model.prefix_encoder.state_dict())
    else:
        for (n_s, p_s), (n_f, p_f) in zip(server_model.named_parameters(), full_model.named_parameters()):
            if "lora" in n_s:
                p_f.data.copy_(p_s.data)


# This function is now replaced by enhanced metrics from metrics.py
# The calculate_enhanced_metrics function provides proper F1 and EM scoring


# KEEP your existing training functions exactly as they are
def train_split_learning_sgd(server_model, full_model, train_loader, eval_loader, optimizer, scheduler, trainer, device, tokenizer, args):
    """Enhanced SGD training with real-time loss reporting"""
    server_model.train()
    total_loss = 0.0
    total_acc = 0.0
    batch_count = 0
    recent_losses = []
    recent_accs = []
    
    # Create progress bar with custom description
    pbar = tqdm(train_loader, desc=f"Training {args.task.upper()}")
    
    for batch in pbar:
        if batch_count >= args.max_steps:
            break
            
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        server_output = server_model(input_ids, attention_mask)
        trainer.send_data({
            'type': 'forward',
            'inputs_embeds': server_output['inputs_embeds'].detach().cpu(),
            'attention_mask': server_output['attention_mask'].detach().cpu(),
            'labels': labels.detach().cpu(),
            'iteration': batch_count,
            'tuning_mode': args.tuning_mode,
            'formatted_text': batch.get('formatted_text', []),
            'original_example': batch.get('original_example', [])
        })
        
        client_response = trainer.receive_data()
        client_loss = client_response['loss'].to(device)
        server_grad_flat = client_response['server_grad'].to(device)
        
        # Calculate metrics using full model for monitoring
        outputs = full_model(input_ids, attention_mask, labels)
        server_loss, acc, f1, em = calculate_enhanced_metrics(outputs, labels, batch, tokenizer, full_model, device, task=args.task)
        
        # Use client loss for training (this is what's actually optimized)
        current_loss = client_loss.item()
        total_loss += current_loss
        total_acc += acc
        
        # Track recent metrics for moving average
        recent_losses.append(current_loss)
        recent_accs.append(acc)
        
        # Keep only last 10 for moving average
        if len(recent_losses) > 10:
            recent_losses.pop(0)
            recent_accs.pop(0)
        
        # Apply gradients
        if server_model.tuning_mode == "prefix":
            grad_shape = server_model.prefix_encoder.weight.shape
            server_grad = server_grad_flat.view(grad_shape)
            server_model.prefix_encoder.weight.grad = server_grad
        else:
            grad_idx = 0
            for param in get_server_trainable_params(server_model):
                param_size = param.numel()
                param.grad = server_grad_flat[grad_idx:grad_idx + param_size].view(param.shape)
                grad_idx += param_size
        
        optimizer.step()
        optimizer.zero_grad()
        batch_count += 1
        
        # Update progress bar with current metrics every batch
        avg_recent_loss = sum(recent_losses) / len(recent_losses)
        avg_recent_acc = sum(recent_accs) / len(recent_accs)
        
        pbar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Avg_Loss': f'{avg_recent_loss:.4f}',
            'Acc': f'{acc:.3f}',
            'Avg_Acc': f'{avg_recent_acc:.3f}'
        })
        
        # Print detailed progress every 50 batches
        if batch_count % 50 == 0:
            print(f"\nStep {batch_count}/{args.max_steps}:")
            print(f"   Current Loss: {current_loss:.6f}, Accuracy: {acc:.6f}")
            print(f"   Average Loss: {avg_recent_loss:.6f}, Accuracy: {avg_recent_acc:.6f}")
            if f1 > 0 or em > 0:
                print(f"   F1: {f1:.6f}, EM: {em:.6f}")
    
    pbar.close()
    scheduler.step()
    return total_loss / batch_count if batch_count > 0 else 0.0, total_acc / batch_count if batch_count > 0 else 0.0


def train_split_learning_zoo(server_model, full_model, train_loader, eval_loader, optimizer, scheduler, trainer, device, tokenizer, args):
    """Enhanced ZOO training with real-time loss reporting"""
    server_model.train()
    zoo = StochasticGradientApproximator(
        get_server_trainable_params(server_model),
        perturbation_scale=args.mu,
        sample_count=args.num_pert,
        compute_device=device
    )
    total_loss = 0.0
    total_acc = 0.0
    batch_count = 0
    recent_losses = []
    recent_accs = []
    
    # Create progress bar with custom description
    pbar = tqdm(train_loader, desc=f"Training {args.task.upper()} (ZOO)")
    
    for batch in pbar:
        if batch_count >= args.max_steps:
            break
            
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        def objective_fn(inputs, targets):
            server_output = server_model(inputs, attention_mask)
            trainer.send_data({
                'type': 'forward',
                'inputs_embeds': server_output['inputs_embeds'].detach().cpu(),
                'attention_mask': server_output['attention_mask'].detach().cpu(),
                'labels': targets.detach().cpu(),
                'iteration': batch_count,
                'tuning_mode': args.tuning_mode,
                'formatted_text': batch.get('formatted_text', []),
                'original_example': batch.get('original_example', [])
            })
            client_response = trainer.receive_data()
            return client_response['loss'].to(device)
        
        zoo.estimate_gradients(input_ids, labels, objective_fn, random_seed=args.seed + batch_count)
        optimizer.step()
        optimizer.zero_grad()
        
        # Calculate metrics and F1 scores for monitoring
        outputs = full_model(input_ids, attention_mask, labels)
        current_loss, acc, f1, em = calculate_enhanced_metrics(outputs, labels, batch, tokenizer, full_model, device, task=args.task)
        
        total_loss += current_loss
        total_acc += acc
        
        # Track recent metrics for moving average
        recent_losses.append(current_loss)
        recent_accs.append(acc)
        
        # Keep only last 10 for moving average
        if len(recent_losses) > 10:
            recent_losses.pop(0)
            recent_accs.pop(0)
        
        batch_count += 1
        
        # Update progress bar with current metrics
        avg_recent_loss = sum(recent_losses) / len(recent_losses)
        avg_recent_acc = sum(recent_accs) / len(recent_accs)
        
        pbar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Avg_Loss': f'{avg_recent_loss:.4f}',
            'Acc': f'{acc:.3f}',
            'Avg_Acc': f'{avg_recent_acc:.3f}'
        })
        
        # Print detailed progress every 50 batches
        if batch_count % 50 == 0:
            print(f"\nZOO Step {batch_count}/{args.max_steps}:")
            print(f"   Current Loss: {current_loss:.6f}, Accuracy: {acc:.6f}")
            print(f"   Average Loss: {avg_recent_loss:.6f}, Accuracy: {avg_recent_acc:.6f}")
            if f1 > 0 or em > 0:
                print(f"   F1: {f1:.6f}, EM: {em:.6f}")
    
    pbar.close()
    scheduler.step()
    return total_loss / batch_count if batch_count > 0 else 0.0, total_acc / batch_count if batch_count > 0 else 0.0


def evaluate_model(model, eval_loader, device, tokenizer, args, task='squad'):
    """Enhanced evaluation with proper F1 and EM scoring"""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    total_em = 0.0
    batch_count = 0
    
    print(f"Starting {task.upper()} evaluation with F1 scoring...")
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Evaluating {task.upper()}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, labels)
            
            # Calculate enhanced metrics with F1 scoring
            loss, acc, f1, em = calculate_enhanced_metrics(
                outputs, labels, batch, tokenizer, model, device, task=task
            )
            
            total_loss += loss
            total_acc += acc
            total_f1 += f1
            total_em += em
            batch_count += 1
            
            # Print progress every 10 batches
            if batch_count % 10 == 0:
                print(f"   Batch {batch_count}: F1={f1:.4f}, EM={em:.4f}, Acc={acc:.4f}")
    
    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    avg_acc = total_acc / batch_count if batch_count > 0 else 0.0
    avg_f1 = total_f1 / batch_count if batch_count > 0 else 0.0
    avg_em = total_em / batch_count if batch_count > 0 else 0.0
    
    print(f"{task.upper()} Evaluation Complete:")
    print(f"   Average Loss: {avg_loss:.6f}")
    print(f"   Average Accuracy: {avg_acc:.6f}")
    print(f"   Average F1 Score: {avg_f1:.6f}")
    print(f"   Average EM Score: {avg_em:.6f}")
    
    return avg_loss, avg_acc, avg_f1, avg_em


# KEEP your existing Trainer class exactly as is
class Trainer:
    def __init__(self, conn):
        self.conn = conn
    
    def send_data(self, data):
        try:
            serialized = pickle.dumps(data)
            self.conn.sendall(len(serialized).to_bytes(4, 'big'))
            self.conn.sendall(serialized)
        except Exception as e:
            print(f"Failed to send data: {e}")
            raise
    
    def receive_data(self):
        try:
            length = int.from_bytes(self.conn.recv(4), 'big')
            data = b''
            while len(data) < length:
                data += self.conn.recv(length - len(data))
            return pickle.loads(data)
        except Exception as e:
            print(f"Failed to receive data: {e}")
            raise


def parse_args():
    """Enhanced argument parser with new dataset support"""
    parser = argparse.ArgumentParser(description='Enhanced Split Learning LLM Server with Multi-Task support')
    parser.add_argument('--model_name', type=str, default='facebook/opt-125m', help='Pretrained model name')
    parser.add_argument('--tuning_mode', type=str, default='prefix', choices=['prefix', 'lora'], help='Fine-tuning method')
    parser.add_argument('--num_prefix', type=int, default=5, help='Number of prefix tokens (prefix mode)')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--zoo_lr', type=float, default=1e-2, help='Learning rate for ZOO')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--train_batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=4, help='Test batch size')
    parser.add_argument('--train_examples', type=int, default=16000, help='Number of training examples')
    parser.add_argument('--eval_examples', type=int, default=3200, help='Number of eval examples')
    parser.add_argument('--max_steps', type=int, default=60000, help='Maximum training steps')
    parser.add_argument('--eval_steps', type=int, default=60000, help='Evaluate every N steps')
    parser.add_argument('--mu', type=float, default=1e-1, help='ZOO perturbation scale')
    parser.add_argument('--num_pert', type=int, default=5, help='ZOO perturbations')
    parser.add_argument('--use_zeroth_order', action='store_true', help='Use ZOO for server')
    parser.add_argument('--use_zeroth_order_client', action='store_true', help='Use ZOO for client')
    parser.add_argument('--evaluate_every', type=int, default=1, help='Evaluate every N epochs')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    
    # NEW: Task selection
    parser.add_argument('--task', type=str, default='squad', 
                       choices=['squad', 'drop', 'sst2'], 
                       help='Dataset task to use')
    
    return parser.parse_args()


if __name__ == "__main__":
    try:
        print("STARTING ENHANCED SPLIT LEARNING SERVER WITH MULTI-TASK SUPPORT")
        print("=" * 80)
        
        args = parse_args()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        print(f"Configuration:")
        print(f"   Model: {args.model_name}")
        print(f"   Task: {args.task.upper()}")
        print(f"   Tuning Mode: {args.tuning_mode.upper()}")
        if args.tuning_mode == "prefix":
            print(f"   Prefix tokens: {args.num_prefix}")
        elif args.tuning_mode == "lora":
            print(f"   LoRA r: {args.lora_r}, alpha: {args.lora_alpha}")
        print(f"   Batch size: {args.train_batch_size}")
        print(f"   Train examples: {args.train_examples}")
        print(f"   Eval examples: {args.eval_examples}")
        print(f"   Max steps: {args.max_steps}")
        print(f"   ZOO server: {args.use_zeroth_order}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        print(f"Loading tokenizer: {args.model_name}")
        tokenizer = safe_get_hf_tokenizer(args.model_name)
        print("Tokenizer loaded successfully")
        
        print(f"Creating models ({args.tuning_mode} mode)...")
        server_model = ServerLLMModel(
            args.model_name, 
            tuning_mode=args.tuning_mode,
            num_prefix=args.num_prefix,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha
        ).to(device)
        
        full_model = FullLLMModel(
            args.model_name,
            tuning_mode=args.tuning_mode,
            num_prefix=args.num_prefix,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha
        ).to(device)
        
        sync_models(server_model, full_model)
        print("Models created and synchronized")
        
        print("Creating dataloaders...")
        train_loader, eval_loader = get_dataloaders(args, tokenizer)
        print("Dataloaders created successfully")
        
        # Rest of your existing main code stays exactly the same...
        print("Setting up optimizer...")
        server_params = get_server_trainable_params(server_model)
        
        if args.use_zeroth_order:
            optimizer = optim.SGD(server_params, lr=args.zoo_lr, momentum=0.0)
        else:
            optimizer = optim.SGD(server_params, lr=args.lr, momentum=args.momentum)
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        print("Optimizer ready")
        
        print("Setting up network...")
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('localhost', 12345))
        server_socket.listen(1)
        
        print("=" * 80)
        print(f"SERVER READY - WAITING FOR CLIENT ({args.task.upper()})")
        print("=" * 80)
        print("Server listening on localhost:12345")
        print(f"Task: {args.task.upper()}")
        print(f"Mode: {args.tuning_mode.upper()}")
        print("Start client with same parameters")
        print("=" * 80)
        
        conn, addr = server_socket.accept()
        print(f"Client connected from {addr}")
        
        trainer = Trainer(conn)
        client_config = trainer.receive_data()
        print(f"Received client config: {client_config}")
        
        print(f"Starting {args.task.upper()} training...")
        
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            if args.use_zeroth_order:
                train_loss, train_acc = train_split_learning_zoo(
                    server_model, full_model, train_loader, eval_loader, optimizer, 
                    scheduler, trainer, device, tokenizer, args
                )
            else:
                train_loss, train_acc = train_split_learning_sgd(
                    server_model, full_model, train_loader, eval_loader, optimizer, 
                    scheduler, trainer, device, tokenizer, args
                )
            
            print(f"Epoch {epoch+1} Training: Loss {train_loss:.4f}, Acc {train_acc:.6f}")
            
            if (epoch + 1) % args.evaluate_every == 0:
                print(f"Running evaluation...")
                sync_models(server_model, full_model)
                eval_loss, eval_acc, eval_f1, eval_em = evaluate_model(
                    full_model, eval_loader, device, tokenizer, args, task=args.task
                )
                
                print(f"\nEPOCH {epoch+1} RESULTS ({args.task.upper()}):")
                print(f"{'='*80}")
                print(f"TRAINING   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
                print(f"EVALUATION - Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.4f}, F1: {eval_f1:.4f}, EM: {eval_em:.4f}")
                print(f"{'='*80}")
        
        trainer.send_data({'type': 'training_complete'})
        print(f"\n{args.task.upper()} training completed successfully!")
        
        # Final evaluation
        print("\nFinal model evaluation...")
        sync_models(server_model, full_model)
        final_loss, final_acc, final_f1, final_em = evaluate_model(
            full_model, eval_loader, device, tokenizer, args, task=args.task
        )
        
        print(f"\nFINAL RESULTS:")
        print(f"{'='*80}")
        print(f"Model: {args.model_name}")
        print(f"Task: {args.task.upper()}")
        print(f"Tuning Mode: {args.tuning_mode.upper()}")
        print(f"Epochs: {args.epochs}")
        print(f"Optimization: {'ZOO' if args.use_zeroth_order else 'SGD'}")
        print(f"Final Loss: {final_loss:.4f}")
        print(f"Final Accuracy: {final_acc:.4f}")
        print(f"Trainable Parameters: {sum(p.numel() for p in server_params):,}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"CRITICAL SERVER ERROR: {e}")
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        try:
            if 'conn' in locals():
                conn.close()
            if 'server_socket' in locals():
                server_socket.close()
        except:
            pass
        print("Server shutdown complete")