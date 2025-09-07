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
from SGDGradientEst import StochasticGradientApproximator

from prefix_kv import PrefixKV, load_grad_state_into

class ServerKVOnly(nn.Module):
    """
    Minimal server-side holder for KV prefixes on the first `cut_layer` layers.
    (No forward compute here to keep changes minimal; client runs the full model and uses these prefixes.)
    """
    def __init__(self, model_name, cut_layer, num_prefix=10):
        super().__init__()
        # load config to size params correctly
        tmp = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map=None)
        self.total_layers = tmp.config.num_hidden_layers
        self.cut_layer = cut_layer
        self.kv = PrefixKV(tmp.config, list(range(0, cut_layer)), num_prefix=num_prefix, device=tmp.device)
        # we do not keep the full model in memory here to save RAM
        del tmp
        torch.cuda.empty_cache()

    def state_dict_kv(self):
        # minimal state dict to send to client
        return {"k": self.kv.k.detach().cpu(), "v": self.kv.v.detach().cpu()}

from metrics import (
    calculate_squad_metrics,
    calculate_generation_f1_em, 
    test_generation_simple,
    normalize_answer,
    squad_f1_score,
    squad_exact_match
)


def squad_collate_fn(batch):
    """Custom collate function for SQUAD dataset with mixed data types"""
    try:
        # Separate tensor and non-tensor data
        input_ids = []
        attention_masks = []
        labels = []
        formatted_texts = []
        original_examples = []
        
        for item in batch:
            input_ids.append(item['input_ids'])
            attention_masks.append(item['attention_mask'])
            labels.append(item['labels'])
            
            # Handle optional fields
            if 'formatted_text' in item:
                formatted_texts.append(item['formatted_text'])
            else:
                formatted_texts.append("")  # Default empty string
                
            if 'original_example' in item:
                original_examples.append(item['original_example'])
            else:
                original_examples.append({})  # Default empty dict
        
        # Stack tensors
        batch_dict = {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks),
            'labels': torch.stack(labels),
        }
        
        # Add non-tensor data only if we have valid data
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


def safe_get_hf_tokenizer(model_name):
    """Safe tokenizer loading with error handling"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        print(f"Failed to load tokenizer for {model_name}: {e}")
        raise


class TextDataset(Dataset):
    """Simple text dataset for demonstration purposes"""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        try:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            
            return {
                'input_ids': input_ids,
                        'server_kv_state': server_model.state_dict_kv(),
                        'cut_layer': args.cut_layer,
                        'server_kv_state': server_model.state_dict_kv(),
                        'cut_layer': args.cut_layer,
                'attention_mask': attention_mask,
                'labels': input_ids.clone()
            }
        except Exception as e:
            print(f"Error processing text at index {idx}: {e}")
            # Return a dummy sample
            dummy_ids = torch.zeros(self.max_length, dtype=torch.long)
            return {
                'input_ids': dummy_ids,
                'attention_mask': torch.ones_like(dummy_ids),
                'labels': dummy_ids.clone()
            }


def get_squad_dataloaders(args, tokenizer):
    """Create SQUAD dataloaders with custom collate function"""
    print(f"Creating SQUAD dataset with MeZO hyperparameters...")
    print(f"Train examples: {args.train_examples}")
    print(f"Dev examples: {args.dev_examples}")
    print(f"Eval examples: {args.eval_examples}")
    print(f"Batch size: {args.train_batch_size}")
    
    try:
        from datasets import load_dataset
        
        # Load SQUAD dataset
        dataset = load_dataset('squad')
        
        # Use the specified sizes
        train_size = min(args.train_examples, len(dataset['train']))
        dev_size = min(args.dev_examples, len(dataset['validation']))
        eval_size = min(args.eval_examples, len(dataset['validation']))
        
        # Create datasets with specified sizes
        train_dataset = dataset['train'].shuffle(seed=args.seed).select(range(train_size))
        val_dataset = dataset['validation'].shuffle(seed=args.seed)
        dev_dataset = val_dataset.select(range(dev_size))
        eval_dataset = val_dataset.select(range(dev_size, min(dev_size + eval_size, len(val_dataset))))
        
        print(f"Dataset sizes: Train={len(train_dataset)}, Dev={len(dev_dataset)}, Eval={len(eval_dataset)}")
        
        # Format datasets
        def format_squad_example(example):
            context = example['context']
            question = example['question']
            answer = example['answers']['text'][0] if len(example['answers']['text']) > 0 else ""
            return f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
        
        # Create formatted texts
        train_texts = [format_squad_example(ex) for ex in train_dataset]
        eval_examples = list(eval_dataset)  # Keep original for evaluation
        eval_texts = [format_squad_example(ex) for ex in eval_examples]
        
        # Create datasets
        train_squad_dataset = SQuADDataset(train_texts, tokenizer, args.max_length)
        eval_squad_dataset = SQuADDataset(eval_texts, tokenizer, args.max_length, eval_examples)
        
        # Create dataloaders with custom collate function
        train_loader = DataLoader(
            train_squad_dataset, 
            batch_size=args.train_batch_size, 
            shuffle=True,
            collate_fn=squad_collate_fn  # Use custom collate function
        )
        eval_loader = DataLoader(
            eval_squad_dataset, 
            batch_size=args.test_batch_size, 
            shuffle=False,
            collate_fn=squad_collate_fn  # Use custom collate function
        )
        
        print(f"SQUAD dataloaders created successfully with custom collate function")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Eval batches: {len(eval_loader)}")
        
        # Test the dataloader
        print("  Testing dataloader...")
        try:
            test_batch = next(iter(train_loader))
            print(f"Dataloader test passed")
            print(f"   Batch keys: {list(test_batch.keys())}")
            print(f"   Batch shapes: {[f'{k}: {v.shape if torch.is_tensor(v) else len(v)}' for k, v in test_batch.items()]}")
        except Exception as test_error:
            print(f"❌ Dataloader test failed: {test_error}")
            raise
        
        return train_loader, eval_loader
        
    except Exception as e:
        print(f"❌ SQUAD dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


class SQuADDataset(Dataset):
    """SQUAD dataset with consistent output structure"""
    def __init__(self, texts, tokenizer, max_length=512, original_examples=None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.original_examples = original_examples or []
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        try:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            
            # Ensure consistent tensor shapes
            if input_ids.dim() == 0:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask.dim() == 0:
                attention_mask = attention_mask.unsqueeze(0)
            
            # For training, labels = input_ids (next token prediction)
            labels = input_ids.clone()
            
            result = {
                'input_ids': input_ids,
                        'server_kv_state': server_model.state_dict_kv(),
                        'cut_layer': args.cut_layer,
                        'server_kv_state': server_model.state_dict_kv(),
                        'cut_layer': args.cut_layer,
                'attention_mask': attention_mask,
                'labels': labels
            }
            
            # Add optional fields consistently
            result['formatted_text'] = text  # Always include text
            
            # Add original example if available
            if idx < len(self.original_examples):
                result['original_example'] = self.original_examples[idx]
            else:
                result['original_example'] = {}  # Empty dict as placeholder
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing item {idx}: {e}")
            # Return a dummy item with consistent structure
            dummy_ids = torch.zeros(self.max_length, dtype=torch.long)
            return {
                'input_ids': dummy_ids,
                'attention_mask': torch.ones_like(dummy_ids),
                'labels': dummy_ids.clone(),
                'formatted_text': "",
                'original_example': {}
            }


class PrefixEncoder(nn.Module):
    """Prefix encoder that creates trainable prefix embeddings"""
    def __init__(self, config, num_prefix=5):
        super(PrefixEncoder, self).__init__()
        self.num_prefix = num_prefix
        self.hidden_size = config.hidden_size
        
        # FIXED: Better initialization - use same as model embeddings
        self.prefix_embeddings = nn.Parameter(
            torch.randn(num_prefix, self.hidden_size) * (self.hidden_size ** -0.5)
        )
        
        # Initialize to match existing embedding statistics
        with torch.no_grad():
            # Initialize with normal distribution similar to model embeddings
            nn.init.normal_(self.prefix_embeddings, mean=0.0, std=0.02)
        
        print(f"  PrefixEncoder: {num_prefix} tokens x {self.hidden_size} dims = {num_prefix * self.hidden_size} parameters")
        print(f"  Prefix embedding std: {self.prefix_embeddings.std().item():.6f}")
    
    def forward(self, batch_size):
        """Expand prefix embeddings for the given batch size"""
        return self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)


class ServerLLMModel(nn.Module):
    """Server-side model that only trains prefix embeddings"""
    def __init__(self, model_name, num_prefix=5):
        super(ServerLLMModel, self).__init__()
        print(f"Loading server model: {model_name}")
        
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float32,
                device_map=None
            )
            print(f"✅ Base model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load base model: {e}")
            raise
        
        # Freeze base model parameters - server only trains prefix
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Create trainable prefix encoder
        try:
            self.num_prefix = num_prefix
            print(f"Prefix encoder created successfully")
        except Exception as e:
            print(f"❌ Failed to create prefix encoder: {e}")
            raise
        
        print(f"Base parameters: {sum(p.numel() for p in self.base_model.parameters()):,}")
        
    def forward(self, input_ids, attention_mask):
        """Forward pass that combines prefix with input embeddings"""
        try:
            batch_size, seq_len = input_ids.shape
            
            # Get input embeddings from base model
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
            
            # Get prefix embeddings for this batch
            
            # Concatenate prefix embeddings with input embeddings
            # Shape: [batch_size, num_prefix + seq_len, hidden_size]
            inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)
            
            # Extend attention mask to include prefix tokens
            prefix_mask = torch.ones(batch_size, self.num_prefix, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            
            return {
                'inputs_embeds': inputs_embeds,
                'attention_mask': attention_mask
            }
        except Exception as e:
            print(f"❌ Server model forward pass failed: {e}")
            raise


# class FullLLMModel(nn.Module):
#     """Complete model for evaluation and ZOO training"""
#     def __init__(self, model_name, num_prefix=5):
#         super(FullLLMModel, self).__init__()
#         try:
#             self.base_model = AutoModelForCausalLM.from_pretrained(
#                 model_name,
#                 torch_dtype=torch.float32,
#                 device_map=None
#             )
#             self.num_prefix = num_prefix
            
#             # Freeze base model parameters
#             for param in self.base_model.parameters():
#                 param.requires_grad = False
                
#             print(f"Full model created successfully")
#         except Exception as e:
#             print(f"❌ Failed to create full model: {e}")
#             raise
    
#     def forward(self, input_ids, attention_mask, labels=None):
#         """Complete forward pass through the model"""
#         try:
#             batch_size, seq_len = input_ids.shape
            
#             # Get embeddings
#             inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
            
#             # Concatenate prefix with input embeddings
#             inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)
            
#             # Extend attention mask for prefix tokens
#             prefix_mask = torch.ones(batch_size, self.num_prefix, device=attention_mask.device, dtype=attention_mask.dtype)
#             attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            
#             # Handle labels for prefix tokens
#             if labels is not None:
#                 # Prefix tokens don't predict next tokens, so label = -100 (ignored)
#                 prefix_labels = torch.full((batch_size, self.num_prefix), -100, device=labels.device, dtype=labels.dtype)
#                 labels = torch.cat([prefix_labels, labels], dim=1)
                
#                 # Ensure length consistency between inputs and labels
#                 input_len = inputs_embeds.shape[1]
#                 if labels.shape[1] != input_len:
#                     if labels.shape[1] > input_len:
#                         labels = labels[:, :input_len]
#                     else:
#                         pad_len = input_len - labels.shape[1]
#                         padding = torch.full((batch_size, pad_len), -100, device=labels.device, dtype=labels.dtype)
#                         labels = torch.cat([labels, padding], dim=1)
            
#             # Forward through base model
#             outputs = self.base_model(
#                 inputs_embeds=inputs_embeds,
#                 attention_mask=attention_mask,
#                 labels=labels,
#                 use_cache=False
#             )
            
#             return outputs
#         except Exception as e:
#             print(f"❌ Full model forward pass failed: {e}")
#             raise
    
#     def generate(self, input_ids, attention_mask, **kwargs):
#         """Generation method for evaluation"""
#         try:
#             batch_size, seq_len = input_ids.shape
            
#             # Get embeddings with prefix
#             inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
#             inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)
            
#             # Extend attention mask
#             prefix_mask = torch.ones(batch_size, self.num_prefix, device=attention_mask.device, dtype=attention_mask.dtype)
#             attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            
#             # Generate
#             return self.base_model.generate(
#                 inputs_embeds=inputs_embeds,
#                 attention_mask=attention_mask,
#                 **kwargs
#             )
#         except Exception as e:
#             print(f"❌ Generation failed: {e}")
#             raise


class FullLLMModel(nn.Module):
    """Frozen full model used only for monitoring/evaluation."""
    def __init__(self, model_name, num_prefix=5):
        super(FullLLMModel, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map=None
        )
        self.num_prefix = num_prefix
        for p in self.base_model.parameters():
            p.requires_grad = False
        print("Full model created successfully (no prefix concat during eval)")

    def forward(self, input_ids, attention_mask, labels=None):
        # No prefix concat; just evaluate the frozen base model for a stable loss/metric.
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False
        )

    def generate(self, input_ids, attention_mask, **kwargs):
        # Safe generation without manual prefix handling
        return self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )



class Trainer:
    """Handles communication with the client"""
    def __init__(self, conn):
        self.conn = conn
    
    def send_data(self, data):
        try:
            serialized = pickle.dumps(data)
            self.conn.sendall(len(serialized).to_bytes(4, 'big'))
            self.conn.sendall(serialized)
        except Exception as e:
            print(f"❌ Failed to send data: {e}")
            raise
    
    def receive_data(self):
        try:
            length = int.from_bytes(self.conn.recv(4), 'big')
            data = b''
            while len(data) < length:
                data += self.conn.recv(length - len(data))
            return pickle.loads(data)
        except Exception as e:
            print(f"❌ Failed to receive data: {e}")
            raise


def calculate_metrics(outputs, labels, batch, tokenizer, model, device):
    """Calculate SQUAD-specific metrics - ROBUST VERSION"""
    try:
        # 1. Calculate standard next-token loss
        loss = outputs.loss.item() if outputs.loss is not None else 0.0
        
        # 2. Calculate token-level accuracy (for monitoring)
        logits = outputs.logits
        if logits.shape[1] != labels.shape[1]:
            min_len = min(logits.shape[1], labels.shape[1])
            logits = logits[:, :min_len, :]
            labels = labels[:, :min_len]
        
        # For next token prediction
        if logits.shape[1] > 1:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
        else:
            shift_logits = logits
            shift_labels = labels
        
        predictions = torch.argmax(shift_logits, dim=-1)
        
        # Calculate answer token accuracy
        answer_accuracy = calculate_answer_token_accuracy(
            predictions, shift_labels, batch, tokenizer
        )
        
        # 3. Calculate F1/EM by generating answers (with error handling)
        f1_score = 0.0
        em_score = 0.0
        
        # Only try generation if we have the required data
        if ('original_example' in batch and 'formatted_text' in batch and 
            len(batch.get('original_example', [])) > 0 and 
            len(batch.get('formatted_text', [])) > 0):
            
            try:
                f1_score, em_score = calculate_generation_f1_em(
                    model, batch, tokenizer, device
                )
            except Exception as gen_error:
                print(f"⚠️ Generation metrics failed: {gen_error}")
                f1_score, em_score = 0.0, 0.0
        
        return loss, answer_accuracy, f1_score, em_score
        
    except Exception as e:
        print(f"❌ SQUAD metrics calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0, 0.0, 0.0


def calculate_answer_token_accuracy(predictions, labels, batch, tokenizer):
    """Calculate accuracy only on answer portion tokens"""
    try:
        if 'formatted_text' not in batch:
            # Fallback to general accuracy
            mask = (labels != -100)
            if mask.sum() == 0:
                return 0.0
            correct = (predictions == labels) & mask
            return correct.sum().float() / mask.sum().float()
        
        # Find answer tokens in each example
        accuracies = []
        for i in range(len(batch['formatted_text'])):
            text = batch['formatted_text'][i]
            
            # Find "Answer:" position
            answer_start = text.find("Answer:")
            if answer_start == -1:
                continue
                
            # Tokenize to find answer token positions
            context_question = text[:answer_start + len("Answer:")]
            answer_part = text[answer_start + len("Answer:"):]
            
            context_tokens = tokenizer.encode(context_question, add_special_tokens=False)
            answer_tokens = tokenizer.encode(answer_part, add_special_tokens=False)
            
            if len(answer_tokens) == 0:
                continue
            
            # Get accuracy for answer tokens only
            start_idx = len(context_tokens)
            end_idx = start_idx + len(answer_tokens)
            
            if end_idx <= predictions.shape[1] and end_idx <= labels.shape[1]:
                answer_preds = predictions[i, start_idx:end_idx]
                answer_labels = labels[i, start_idx:end_idx]
                
                if len(answer_preds) > 0:
                    correct = (answer_preds == answer_labels).sum().item()
                    total = len(answer_preds)
                    accuracies.append(correct / total)
        
        return sum(accuracies) / len(accuracies) if accuracies else 0.0
        
    except Exception as e:
        print(f"❌ Answer token accuracy failed: {e}")
        return 0.0


def squad_f1_score(prediction, ground_truth):
    """Calculate F1 score for SQUAD with better error handling"""
    try:
        from collections import Counter
        import string
        import re
        
        def normalize_answer(s):
            """Normalize answer text"""
            if not isinstance(s, str):
                s = str(s)
            
            def remove_articles(text):
                return re.sub(r'\b(a|an|the)\b', ' ', text)
            def white_space_fix(text):
                return ' '.join(text.split())
            def remove_punc(text):
                exclude = set(string.punctuation)
                return ''.join(ch for ch in text if ch not in exclude)
            def lower(text):
                return text.lower()
            
            return white_space_fix(remove_articles(remove_punc(lower(s))))
        
        # Normalize inputs
        pred_normalized = normalize_answer(prediction)
        truth_normalized = normalize_answer(ground_truth)
        
        prediction_tokens = pred_normalized.split()
        ground_truth_tokens = truth_normalized.split()
        
        # Handle empty cases
        if len(prediction_tokens) == 0 and len(ground_truth_tokens) == 0:
            return 1.0
        if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
            return 0.0
        
        # Calculate overlap
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        precision = num_same / len(prediction_tokens)
        recall = num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return float(f1)
        
    except Exception as e:
        print(f"❌ F1 calculation error: {e}")
        return 0.0


def evaluate_model(model, test_loader, device, tokenizer, args):
    """SQUAD-specific evaluation with imported metrics"""
    print("  Starting SQUAD evaluation...")
    
    # Test generation capability first
    print("  Testing generation capability...")
    gen_works = test_generation_simple(model, tokenizer, device)
    print(f"  Generation test result: {'✅ PASS' if gen_works else '❌ FAIL'}")
    
    model.eval()
    total_loss = 0.0
    total_answer_accuracy = 0.0
    total_f1 = 0.0
    total_em = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="SQUAD Evaluation")):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Debug: Check data structure for first batch
                if batch_idx == 0:
                    print(f"\n  First batch debug:")
                    print(f"   Input shape: {input_ids.shape}")
                    print(f"   Has formatted_text: {'formatted_text' in batch}")
                    print(f"   Has original_example: {'original_example' in batch}")
                    if 'formatted_text' in batch:
                        print(f"   Formatted text count: {len(batch['formatted_text'])}")
                        print(f"   Sample: {batch['formatted_text'][0][:100]}...")
                
                # Forward pass
                outputs = model(input_ids, attention_mask, labels)
                
                # Calculate SQUAD metrics using imported function
                loss, answer_acc, f1, em = calculate_squad_metrics(
                    outputs, labels, batch, tokenizer, model, device
                )
                
                total_loss += loss
                total_answer_accuracy += answer_acc
                total_f1 += f1
                total_em += em
                num_batches += 1
                
                # Print progress for first few batches
                if batch_idx < 3:
                    print(f"\nBatch {batch_idx}: Loss={loss:.4f}, Acc={answer_acc:.6f}, F1={f1:.6f}, EM={em:.6f}")
                
            except Exception as e:
                print(f"\n⚠️ Error in evaluation batch {batch_idx}: {e}")
                continue
    
    # Calculate averages
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_answer_accuracy = total_answer_accuracy / num_batches if num_batches > 0 else 0.0
    avg_f1 = total_f1 / num_batches if num_batches > 0 else 0.0
    avg_em = total_em / num_batches if num_batches > 0 else 0.0
    
    print(f"\n  SQUAD Evaluation Complete:")
    print(f"   Average Loss: {avg_loss:.4f}")
    print(f"   Answer Token Accuracy: {avg_answer_accuracy:.6f}")
    print(f"   F1 Score: {avg_f1:.6f}")
    print(f"   Exact Match: {avg_em:.6f}")
    
    return avg_loss, avg_answer_accuracy, avg_f1, avg_em

def train_split_learning_zoo(server_model, full_model, train_loader, eval_loader,
                             optimizer, scheduler, trainer, device, tokenizer, args):
    """Train using split learning with Zeroth-Order Optimization (server-side ZOO)."""
    print("  Starting split learning training (ZOO)...")
    print(f"   ZOO Training configuration:")
    print(f"   Max steps: {args.max_steps}")
    print(f"   Batch size: {args.train_batch_size}")
    print(f"   Learning rate: {args.zoo_lr}")
    print(f"   Perturbation scale (eps): {args.mu}")
    print(f"   Number of perturbations: {args.num_pert}")
    print(f"   Eval every: {args.eval_steps} steps")

    # --- FIX 1: define params to perturb (server KV prefixes only) ---
    server_params = list(server_model.kv.parameters())

    # Create gradient estimator for server KV
    grad_estimator = StochasticGradientApproximator(
        model_params=server_params,
        perturbation_scale=args.mu,
        sample_count=args.num_pert,
        compute_device=device,
        data_type=torch.float32
    )

    server_model.train()
    losses, accs = [], []
    global_step = 0

    try:
        pbar = tqdm(total=args.max_steps, desc="Training (ZOO)",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Loss: {postfix}')

        epoch = 0
        while global_step < args.max_steps:
            epoch += 1
            print(f"\nEpoch {epoch} (Steps {global_step}/{args.max_steps})")

            for batch_idx, batch in enumerate(train_loader):
                if global_step >= args.max_steps:
                    break

                try:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)

                    # Build labels JUST like the SGD path: -100 on pads
                    labels = input_ids.clone()
                    labels[attention_mask == 0] = -100
                    pad_id = getattr(tokenizer, 'pad_token_id', None)
                    if pad_id is not None:
                        labels[input_ids == pad_id] = -100
                    labels = labels.long()

                    optimizer.zero_grad(set_to_none=True)

                    # ---- objective uses *current* (possibly perturbed) KV: compute loss on CLIENT ----
                    def objective_fn(_x=None, _y=None):
                        payload = {
                            'input_ids': input_ids,
                            'attention_mask': attention_mask,
                            'labels': labels,
                            'server_kv_state': server_model.state_dict_kv(),  # <- reads current KV (incl. perturbations)
                            'cut_layer': args.cut_layer,
                            'zoo_eval': True,
                        }
                        # Ask client to run forward and return loss
                        trainer.send_data({'type': 'forward', 'data': payload})
                        client_resp = trainer.receive_data()

                        if 'error' in client_resp:
                            raise RuntimeError(f"Client error during ZOO objective: {client_resp['error']}")

                        # loss is a CPU tensor or float; convert to device tensor
                        return torch.as_tensor(float(client_resp['loss']), device=device)

                    # Estimate gradients for server KV via ZOO
                    pbar.set_postfix_str("Computing ZOO gradients...")
                    try:
                        # Newer API (keyword)
                        grad_estimator.model_params = server_params
                        grad_estimator.estimate_gradients(
                            random_seed=global_step * 1000 + args.seed,
                            objective_fn=objective_fn
                        )
                    except TypeError:
                        # Legacy API (positional)
                        grad_estimator.estimate_gradients(
                            input_ids, labels, objective_fn,
                            random_seed=global_step * 1000 + args.seed
                        )

                    # Apply the ZOO estimated gradients to server KV
                    optimizer.step()

                    # --- monitor (safe) ---
                    try:
                        with torch.no_grad():
                            out = full_model(input_ids, attention_mask, labels)   # frozen full model, no prefixes
                            loss_val, acc, _, _ = calculate_metrics(
                                out, labels, batch, tokenizer, full_model, device
                            )
                    except Exception:
                        # if monitoring hiccups, keep training
                        loss_val, acc = float(objective_fn().item()), 0.0

                    losses.append(loss_val)
                    accs.append(acc)

                    global_step += 1
                    cur_loss = sum(losses[-10:]) / min(len(losses), 10)
                    cur_acc  = sum(accs[-10:]) / min(len(accs), 10)
                    pbar.set_postfix_str(f"{cur_loss:.4f}, Acc: {cur_acc:.6f}")
                    pbar.update(1)

                    # periodic eval (uses the safe FullLLMModel)
                    if global_step % args.eval_steps == 0:
                        print(f"\nStep {global_step}: Running ZOO evaluation...")
                        eval_loss, eval_acc, eval_f1, eval_em = evaluate_model(
                            full_model, eval_loader, device, tokenizer, args
                        )
                        print(f".  Step {global_step} ZOO Evaluation:")
                        print(f"   Loss: {eval_loss:.4f}")
                        print(f"   Answer Accuracy: {eval_acc:.6f}")
                        print(f"   F1 Score: {eval_f1:.6f}")
                        print(f"   Exact Match: {eval_em:.6f}")
                        server_model.train()

                except Exception as e:
                    print(f"\n✖ Error in ZOO step {global_step}: {e}")
                    traceback.print_exc()
                    continue

        pbar.close()
        scheduler.step()

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        avg_acc  = sum(accs) / len(accs) if accs else 0.0
        print(f"\n✅ ZOO Training Complete - Final Loss: {avg_loss:.4f}, Final Acc: {avg_acc:.6f}")
        return avg_loss, avg_acc

    except Exception as e:
        print(f"✖ ZOO training failed: {e}")
        traceback.print_exc()
        return 0.0, 0.0


def train_split_learning_sgd(server_model, full_model, train_loader, eval_loader, optimizer, 
                           scheduler, trainer, device, tokenizer, args):
    """Train using split learning with SGD - FIXED VERSION"""
    print("   Starting split learning training (SGD)...")
    print(f"   Training configuration:")
    print(f"   Max steps: {args.max_steps}")
    print(f"   Batch size: {args.train_batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Eval every: {args.eval_steps} steps")
    
    server_model.train()
    losses = []
    accs = []
    global_step = 0
    
    try:
        pbar = tqdm(total=args.max_steps, desc="Training (SGD)", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Loss: {postfix}')
        
        epoch = 0
        while global_step < args.max_steps:
            epoch += 1
            print(f"\nEpoch {epoch} (Steps {global_step}/{args.max_steps})")

            for i, batch in enumerate(train_loader):
                if global_step >= args.max_steps:
                    break

                try:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    # labels = batch['labels'].to(device)
                    
                    optimizer.zero_grad()
                    
                    # Server forward pass
                    # server_output = server_model(input_ids, attention_mask)
                    # payload = {
                    #     'input_ids': input_ids,                # shape [B, S]
                    #     'attention_mask': attention_mask,      # shape [B, S]
                    #     'labels': labels,                      # include for training; omit for pure inference
                    #     'server_kv_state': server_model.state_dict_kv(),  # {'k': ..., 'v': ...} on CPU
                    #     'cut_layer': args.cut_layer            # e.g., 6
                    # }
                    
                    labels = input_ids.clone()
                    labels[attention_mask == 0] = -100
                    pad_id = getattr(tokenizer, 'pad_token_id', None)
                    if pad_id is not None:
                        labels[input_ids == pad_id] = -100
                    labels = labels.long()

                    payload = {
                        'input_ids': input_ids,                    # [B, S] (Long)
                        'attention_mask': attention_mask,          # [B, S]
                        'labels': labels,                          # [B, S] with -100 masked
                        'server_kv_state': server_model.state_dict_kv(),  # {'k': tensor, 'v': tensor} on CPU
                        'cut_layer': args.cut_layer,               # e.g., 6
                    }
                    trainer.send_data({'type': 'forward', 'data': payload})

                    # Receive gradients from client
                    client_response = trainer.receive_data()

                    if 'error' in client_response:
                        print(f"Client error: {client_response['error']}")
                        pbar.set_postfix_str("Client Error")
                        continue

                    # Loss for logging
                    loss_val = float(client_response['loss'])
                    losses.append(loss_val)

                    # KV-only: client returns structured grads for server prefixes
                    server_grad_state = client_response.get('server_grad_state', None)
                    if server_grad_state is None:
                        raise RuntimeError("Client did not return 'server_grad_state' (KV mode).")

                    # Apply server grads and step
                    optimizer.zero_grad(set_to_none=True)  # ensure clean .grad
                    load_grad_state_into(server_model.kv, server_grad_state, device=device)  # fills .grad of kv.k / kv.v
                    optimizer.step()

                    
                    # Calculate accuracy for monitoring using full model
                    with torch.no_grad():
                        outputs = full_model(input_ids, attention_mask, labels)
                        loss_check, accuracy, f1, _ = calculate_metrics(outputs, labels, batch, tokenizer, full_model, device)
                        losses.append(loss_val)
                        accs.append(accuracy)
                    
                    # Update progress bar with current loss, accuracy, and F1
                    global_step += 1
                    current_loss = sum(losses[-10:]) / min(len(losses), 10)  # Moving average of last 10
                    current_acc = sum(accs[-10:]) / min(len(accs), 10)
                    
                    pbar.set_postfix_str(f"{current_loss:.4f}, Acc: {current_acc:.3f}")
                    pbar.update(1)
                    
                    # Print detailed stats every 20 batches
                    if (i + 1) % 20 == 0:
                        avg_loss = sum(losses) / len(losses)
                        avg_acc = sum(accs) / len(accs)
                        print(f"\nStep {i+1}/{len(train_loader)} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.3f}")

                    if global_step % args.eval_steps == 0:
                        print(f"\nStep {global_step}: Running evaluation...")
                        eval_loss, eval_acc, eval_f1, eval_em = evaluate_model(
                            full_model, eval_loader, device, tokenizer, args
                        )

                        print(f"   Step {global_step} Evaluation:")
                        print(f"   Loss: {eval_loss:.4f}")
                        print(f"   Answer Accuracy: {eval_acc:.6f}")
                        print(f"   F1 Score: {eval_f1:.6f}")
                        print(f"   Exact Match: {eval_em:.6f}")
                        
                        # Return to training mode
                        server_model.train()

                    # Print progress every 100 steps
                    if global_step % 100 == 0:
                        avg_loss = sum(losses[-100:]) / min(len(losses), 100)
                        avg_acc = sum(accs[-100:]) / min(len(accs), 100)
                        print(f"\nStep {global_step}/{args.max_steps} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.6f}")
                    
                except Exception as e:
                    print(f"\nâŒ Error in batch {i}: {e}")
                    pbar.set_postfix_str(f"Batch {i} Error")
                    continue
        
        scheduler.step()
        
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        avg_acc = sum(accs) / len(accs) if accs else 0.0
        
        print(f"\nSGD Training Complete - Final Loss: {avg_loss:.4f}, Final Acc: {avg_acc:.3f}")
        return avg_loss, avg_acc
        
    except Exception as e:
        print(f"âŒ Training failed: {e}")
        return 0.0, 0.0


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Split Learning LLM Server')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='facebook/opt-125m', help='Model name')
    parser.add_argument('--num_prefix', type=int, default=10, help='Number of prefix tokens')
    parser.add_argument('--cut_layer', type=int, default=6, help='Split index: 0..L-1 goes to server; cut..L-1 to client')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    
    # Training parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--zoo_lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate') 
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--train_batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=4, help='Test batch size')

    # Dataset sizes - NEW ARGUMENTS
    parser.add_argument('--train_examples', type=int, default=1000, help='Number of training examples')  # TRAIN=1000
    parser.add_argument('--dev_examples', type=int, default=500, help='Number of dev examples')  # DEV=500
    parser.add_argument('--eval_examples', type=int, default=1000, help='Number of eval examples')  # EVAL=1000
    
    # Training steps - NEW ARGUMENTS
    parser.add_argument('--max_steps', type=int, default=4000, help='Maximum training steps')  # STEPS=4000
    parser.add_argument('--eval_steps', type=int, default=4000, help='Evaluate every N steps')  # EVAL_STEPS=4000
    
    
    # ZOO parameters
    parser.add_argument('--mu', type=float, default=1e-1, help='ZOO perturbation scale')
    parser.add_argument('--num_pert', type=int, default=5, help='ZOO perturbations')
    parser.add_argument('--use_zeroth_order', action='store_true', help='Use ZOO for server')
    parser.add_argument('--use_zeroth_order_client', action='store_true', help='Use ZOO for client')
    
    # Evaluation
    parser.add_argument('--evaluate_every', type=int, default=1, help='Evaluate every N epochs')
    parser.add_argument('--f1_method', type=str, default='micro', 
                       choices=['micro', 'macro', 'sequence'],
                       help='F1 score calculation method')
    
    return parser.parse_args()


if __name__ == "__main__":
    try:
        print("STARTING ENHANCED SPLIT LEARNING SERVER")
        print("=" * 60)
        
        args = parse_args()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        print(f"  Configuration:")
        print(f"   Model: {args.model_name}")
        print(f"   Batch size: {args.train_batch_size}")
        print(f"   Max length: {args.max_length}")
        print(f"   ZOO server: {args.use_zeroth_order}")
        print(f"   ZOO client: {args.use_zeroth_order_client}")
        print(f"   F1 method: {args.f1_method}")
        if args.use_zeroth_order:
            print(f"   ZOO mu: {args.mu}")
            print(f"   ZOO perturbations: {args.num_pert}")
        
        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Using device: {device}")
        
        # Load tokenizer
        print(f"  Loading tokenizer: {args.model_name}")
        tokenizer = safe_get_hf_tokenizer(args.model_name)
        print("  Tokenizer loaded successfully")
        
        # Create models
        print("  Creating models...")
        server_model = ServerKVOnly(args.model_name, cut_layer=args.cut_layer, num_prefix=args.num_prefix).to(device)
        full_model = FullLLMModel(args.model_name, args.num_prefix).to(device)
        
        # Synchronize models
        print("  Models created and synchronized")
        
        # Create data loaders
        print(" Creating dataloaders...")
        train_loader, eval_loader = get_squad_dataloaders(args, tokenizer)
        print("  Dataloaders created successfully")
        # Setup optimizer
        print("  Setting up optimizer...")
        if args.use_zeroth_order:
            # ZOO needs higher learning rate and no momentum
            optimizer = optim.SGD(server_model.kv.parameters(), 
                                lr=args.zoo_lr, momentum=0.0)
        else:
            # Regular SGD can use lower learning rate with momentum
            optimizer = optim.SGD(server_model.kv.parameters(), 
                                lr=args.lr, momentum=args.momentum)
        
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        print("✅ Optimizer ready")
        
        # Setup network
        print("Setting up network...")
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('localhost', 12345))
        server_socket.listen(1)
        
        # Debug: Print model dimensions
        print(f"  Debug - Server model info:")
        
        print("=" * 60)
        print("SERVER READY - WAITING FOR CLIENT")
        print("=" * 60)
        print("Server listening on localhost:12345")
        print("Start client with same parameters")
        print("=" * 60)
        
        # Accept client connection
        conn, addr = server_socket.accept()
        print(f"✅ Client connected from {addr}")
        
        trainer = Trainer(conn)
        client_config = trainer.receive_data()
        print(f"Received client config: {client_config}")
        
        print("Starting training...")
        
        # Training loop
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            # Choose training method based on SERVER configuration only
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
            
            print(f"✅ Epoch {epoch+1} Training: Loss {train_loss:.4f}, Acc {train_acc:.6f}")
            
            # Evaluation
            if (epoch + 1) % args.evaluate_every == 0:
                print(f"Running evaluation...")
                # eval_loss, eval_acc, eval_f1 = evaluate_model(full_model, test_loader, device, tokenizer, args)
                eval_loss, eval_acc, eval_f1, eval_em = evaluate_model(full_model, eval_loader, device, tokenizer, args)

                
                print(f"\nEPOCH {epoch+1} RESULTS:")
                print(f"{'='*60}")
                print(f"TRAINING   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
                print(f"EVALUATION - Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.4f}, F1: {eval_f1:.4f}")
                print(f"{'='*60}")
        
        # Send training completion signal to client
        trainer.send_data({'type': 'training_complete'})
        print("\nTraining completed successfully!")
        
        # Final evaluation
        print("\nFinal model evaluation...")
        # final_loss, final_acc, final_f1 = evaluate_model(full_model, test_loader, device, tokenizer, args)
        final_loss, final_acc, final_f1, eval_em = evaluate_model(full_model, eval_loader, device, tokenizer, args)

        
        print(f"\nFINAL RESULTS:")
        print(f"{'='*60}")
        print(f"Model: {args.model_name}")
        print(f"Epochs: {args.epochs}")
        print(f"Optimization: {'ZOO' if args.use_zeroth_order else 'SGD'}")
        print(f"Final Loss: {final_loss:.4f}")
        print(f"Final Accuracy: {final_acc:.4f}")
        print(f"Final F1 Score: {final_f1:.4f}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"❌ CRITICAL SERVER ERROR: {e}")
        print("  Full traceback:")
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
        print("  Server shutdown complete")