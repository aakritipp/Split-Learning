import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import re
from collections import Counter
import string


class SQuADDataset(Dataset):
    """
    KEEP YOUR EXISTING SQUAD DATASET CLASS EXACTLY AS IS
    This is your working version - don't change it!
    """
    def __init__(self, split='train', tokenizer=None, max_length=512, num_examples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load SQuAD dataset from HuggingFace
        print(f"Loading SQuAD {split} dataset...")
        dataset = load_dataset('squad', split=split)
        
        if num_examples:
            dataset = dataset.select(range(min(num_examples, len(dataset))))
        
        self.examples = []
        print(f"Processing {len(dataset)} examples...")
        
        for example in dataset:
            context = example['context']
            question = example['question']
            
            # Handle multiple answers (take the first one)
            if isinstance(example['answers']['text'], list) and len(example['answers']['text']) > 0:
                answer = example['answers']['text'][0]
            else:
                answer = ""
            
            # Format as natural language prompt (MeZO style)
            # This follows the pattern mentioned in MeZO paper
            formatted_text = f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
            
            self.examples.append({
                'text': formatted_text,
                'context': context,
                'question': question,
                'answer': answer,
                'id': example['id'],
                'formatted_text': formatted_text,  # Add this for consistency
                'original_example': example  # Add this for metrics
            })
        
        print(f"Loaded {len(self.examples)} SQuAD examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example['text']
        
        # Tokenize the full text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze(),  # For language modeling
            'example_id': example['id'],
            'answer': example['answer'],
            'formatted_text': example['formatted_text'],
            'original_example': example['original_example']
        }


class MultiTaskDataset(Dataset):
    """
    NEW: Multi-task dataset that maintains SQuAD compatibility
    """
    def __init__(self, task='squad', split='train', tokenizer=None, max_length=512, num_examples=None):
        self.task = task
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Loading {task.upper()} {split} dataset...")
        
        if task == 'squad':
            # Use existing SQuAD logic - don't change!
            dataset = load_dataset('squad', split=split)
        elif task == 'drop':
            dataset = load_dataset('drop', split=split)
        elif task == 'xsum':
            # Fix XSum dataset loading to follow reference pattern
            try:
                # Try the standard XSum dataset first
                dataset = load_dataset('EdinburghNLP/xsum', split='train' if split == 'train' else 'test')
            except:
                try:
                    # Try alternative XSum loading
                    dataset = load_dataset('xsum', split='train' if split == 'train' else 'test', revision='main')
                except:
                    # Last resort: try CNN/DailyMail
                    print(f"Warning: XSum loading failed, using CNN/DailyMail as fallback")
                    dataset = load_dataset('cnn_dailymail', '3.0.0', split='train' if split == 'train' else 'test')
        elif task == 'sst2':
            dataset = load_dataset('glue', 'sst2', split='train' if split == 'train' else 'validation')
        else:
            raise ValueError(f"Task {task} not supported")
        
        if num_examples:
            dataset = dataset.select(range(min(num_examples, len(dataset))))
        
        self.examples = []
        print(f"Processing {len(dataset)} {task} examples...")
        
        for example in dataset:
            formatted_example = self._format_example(example, task)
            self.examples.append(formatted_example)
        
        print(f"Loaded {len(self.examples)} {task} examples")
    
    def _format_example(self, example, task):
        """Format example based on task - maintains SQuAD format structure"""
        if task == 'squad':
            # EXACT same formatting as your working version
            context = example['context']
            question = example['question']
            
            if isinstance(example['answers']['text'], list) and len(example['answers']['text']) > 0:
                answer = example['answers']['text'][0]
            else:
                answer = ""
            
            formatted_text = f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
            
            return {
                'text': formatted_text,
                'context': context,
                'question': question,
                'answer': answer,
                'id': example['id'],
                'formatted_text': formatted_text,
                'original_example': example
            }
            
        elif task == 'drop':
            # Similar format to SQuAD
            passage = example['passage']
            question = example['question']
            
            # DROP has different answer structure - debug first few examples
            if len(self.examples) < 3:
                print(f"DEBUG DROP example {len(self.examples)}: keys = {list(example.keys())}")
                if 'answers_spans' in example:
                    print(f"  answers_spans keys: {list(example['answers_spans'].keys())}")
                    print(f"  answers_spans content: {example['answers_spans']}")
            
            # Extract answer from DROP format
            answer = ""
            if 'answers_spans' in example and 'spans' in example['answers_spans']:
                spans = example['answers_spans']['spans']
                if isinstance(spans, list) and len(spans) > 0:
                    answer = spans[0]  # Take first span
            elif 'answer' in example:
                # Some DROP examples might have direct answer field
                answer = example['answer']
            
            formatted_text = f"Context: {passage}\nQuestion: {question}\nAnswer: {answer}"
            
            return {
                'text': formatted_text,
                'context': passage,
                'question': question,
                'answer': answer,
                'id': example.get('query_id', ''),
                'formatted_text': formatted_text,
                'original_example': example
            }
            
        elif task == 'xsum':
            # Handle XSum format following reference implementation pattern
            # XSum: 'document' and 'summary' fields
            # CNN/DailyMail fallback: 'article' and 'highlights' fields
            
            if 'document' in example:
                # Standard XSum format
                document = example['document']
                summary = example['summary']
            elif 'article' in example:
                # CNN/DailyMail fallback format
                document = example['article'] 
                summary = example['highlights']
            else:
                # Debug unknown format
                if len(self.examples) < 3:
                    print(f"DEBUG XSum example {len(self.examples)}: keys = {list(example.keys())}")
                raise ValueError(f"Unknown XSum format - available keys: {list(example.keys())}")
            
            # Use simple Document/Summary format like reference templates
            formatted_text = f"Document: {document}\nSummary: {summary}"
            
            return {
                'text': formatted_text,
                'context': document,
                'question': '',
                'answer': summary,
                'id': example.get('id', example.get('idx', '')),
                'formatted_text': formatted_text,
                'original_example': example
            }
            
        elif task == 'sst2':
            sentence = example['sentence']
            label = example['label']  # 0 = negative, 1 = positive
            
            # Use completion format inspired by reference templates
            # This is much more natural for language models than classification format
            verbalizer = {0: "terrible", 1: "great"}  # Maps labels to completion words
            completion_word = verbalizer[label]
            
            # Format: "Sentence It was word" instead of "Sentence: X\nSentiment: label"
            formatted_text = f"{sentence} It was {completion_word}"
            
            return {
                'text': formatted_text,
                'context': sentence,
                'question': '',
                'answer': completion_word,  # Use completion word as answer
                'id': example.get('idx', ''),
                'formatted_text': formatted_text,
                'original_example': example  # Keep original for ground truth mapping
            }
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example['text']
        
        # Same tokenization as your working SQuAD version
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Return SAME structure as working SQuAD version
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze(),
            'example_id': example['id'],
            'answer': example['answer'],
            'formatted_text': example['formatted_text'],
            'original_example': example['original_example']
        }


# KEEP your existing functions exactly as they are
def normalize_answer(s):
    """Normalize answer for F1 computation (from SQuAD evaluation script)"""
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


def f1_score(prediction, ground_truth):
    """Compute F1 score between prediction and ground truth"""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return int(prediction_tokens == ground_truth_tokens)
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def exact_match_score(prediction, ground_truth):
    """Compute exact match score"""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


# NEW: Enhanced dataloader function that maintains backward compatibility
def get_enhanced_dataloaders(task='squad', tokenizer=None, train_batch_size=2, test_batch_size=2, 
                           max_length=512, num_train_examples=1000, num_eval_examples=200):
    """
    Enhanced version that supports multiple tasks but maintains SQuAD compatibility
    """
    if task == 'squad':
        # Use your existing working SQuAD logic
        train_dataset = SQuADDataset('train', tokenizer, max_length, num_train_examples)
        test_dataset = SQuADDataset('validation', tokenizer, max_length, num_eval_examples)
    else:
        # Use new multi-task dataset for other tasks
        train_dataset = MultiTaskDataset(task, 'train', tokenizer, max_length, num_train_examples)
        test_dataset = MultiTaskDataset(task, 'validation', tokenizer, max_length, num_eval_examples)
    
    # Use your existing collate function
    def squad_collate_fn(batch):
        input_ids = []
        attention_masks = []
        labels = []
        formatted_texts = []
        original_examples = []
        
        for item in batch:
            input_ids.append(item['input_ids'])
            attention_masks.append(item['attention_mask'])
            labels.append(item['labels'])
            formatted_texts.append(item['formatted_text'])
            original_examples.append(item['original_example'])
        
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks),
            'labels': torch.stack(labels),
            'formatted_text': formatted_texts,
            'original_example': original_examples
        }
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        shuffle=True,
        collate_fn=squad_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=test_batch_size, 
        shuffle=False,
        collate_fn=squad_collate_fn
    )
    
    return train_loader, test_loader


# KEEP this function for backward compatibility with your working version
def get_squad_dataloaders(tokenizer, train_batch_size=2, test_batch_size=2, 
                         max_length=512, num_train_examples=1000, num_eval_examples=200):
    """
    Your existing working function - don't change this!
    """
    return get_enhanced_dataloaders('squad', tokenizer, train_batch_size, test_batch_size,
                                   max_length, num_train_examples, num_eval_examples)


def get_hf_tokenizer(model_name):
    """Get HuggingFace tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


if __name__ == "__main__":
    # Test that SQuAD still works exactly as before
    print("Testing SQuAD compatibility...")
    tokenizer = get_hf_tokenizer('gpt2')
    
    # Test with your existing function
    train_loader, test_loader = get_squad_dataloaders(
        tokenizer, train_batch_size=2, test_batch_size=2, max_length=128,
        num_train_examples=10, num_eval_examples=5
    )
    
    print(f"SQuAD - Train: {len(train_loader)}, Test: {len(test_loader)}")
    
    # Test sample batch
    for batch in train_loader:
        print(f"Batch keys: {list(batch.keys())}")
        print(f"Batch shapes: {[f'{k}: {v.shape if torch.is_tensor(v) else len(v)}' for k, v in batch.items()]}")
        print(f"Sample text: {batch['formatted_text'][0][:100]}...")
        break
    
    # Test new datasets
    print("\nTesting new datasets...")
    for task in ['drop', 'xsum', 'sst2']:
        try:
            print(f"\nTesting {task}...")
            train_loader, test_loader = get_enhanced_dataloaders(
                task, tokenizer, train_batch_size=1, test_batch_size=1, max_length=128,
                num_train_examples=2, num_eval_examples=2
            )
            print(f"{task} - Train: {len(train_loader)}, Test: {len(test_loader)}")
            
            # Test batch structure is same as SQuAD
            for batch in train_loader:
                print(f"{task} batch keys: {list(batch.keys())}")
                print(f"Sample text: {batch['formatted_text'][0][:100]}...")
                break
        except Exception as e:
            print(f"{task} failed: {e}")