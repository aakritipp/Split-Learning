import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import re
from collections import Counter
import string
from enum import Enum
from functools import cached_property
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, CliImplicitFlag, SettingsConfigDict


class LmGenerationTask(Enum):
    squad = "squad"
    drop = "drop" 
    xsum = "xsum"


class LmClassificationTask(Enum):
    sst2 = "sst2"


# Dataset mapping for different tasks
LM_DATASET_MAP = {
    "squad": "squad",
    "drop": "drop", 
    "xsum": "xsum",
    "sst2": "glue"
}


class BaseTemplate:
    """Base template class for different tasks"""
    def verbalize(self, example):
        raise NotImplementedError
    
    def encode(self, example):
        raise NotImplementedError


class SQuADTemplate(BaseTemplate):
    def verbalize(self, example):
        context = example['context']
        question = example['question']
        if isinstance(example['answers']['text'], list) and len(example['answers']['text']) > 0:
            answer = example['answers']['text'][0]
        else:
            answer = ""
        return f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
    
    def encode(self, example):
        context = example['context']
        question = example['question']
        return f"Context: {context}\nQuestion: {question}\nAnswer:"


class DropTemplate(BaseTemplate):
    def verbalize(self, example):
        passage = example['passage']
        question = example['question']
        if isinstance(example['answers_spans']['spans'], list) and len(example['answers_spans']['spans']) > 0:
            answer = example['answers_spans']['spans'][0]
        else:
            answer = ""
        return f"Passage: {passage}\nQuestion: {question}\nAnswer: {answer}"
    
    def encode(self, example):
        passage = example['passage']
        question = example['question']
        return f"Passage: {passage}\nQuestion: {question}\nAnswer:"


class XSumTemplate(BaseTemplate):
    def verbalize(self, example):
        document = example['document']
        summary = example['summary']
        return f"Document: {document}\nSummary: {summary}"
    
    def encode(self, example):
        document = example['document']
        return f"Document: {document}\nSummary:"


class SST2Template(BaseTemplate):
    def verbalize(self, example):
        sentence = example['sentence']
        label = "positive" if example['label'] == 1 else "negative"
        return f"Sentence: {sentence}\nSentiment: {label}"
    
    def encode(self, example):
        sentence = example['sentence']
        return f"Sentence: {sentence}\nSentiment:"


# Template mapping
LM_TEMPLATE_MAP = {
    "squad": SQuADTemplate,
    "drop": DropTemplate,
    "xsum": XSumTemplate,
    "sst2": SST2Template
}


class CustomLMDataset(Dataset):
    """Dataset for language modeling tasks (classification/generation)"""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
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
            'labels': encoding['input_ids'].squeeze()
        }


class CustomLMGenerationDataset(Dataset):
    """Dataset specifically for generation tasks with separate inputs and gold outputs"""
    def __init__(self, input_texts, gold_outputs, tokenizer, max_length=512):
        self.input_texts = input_texts
        self.gold_outputs = gold_outputs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.input_texts)
    
    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        gold_output = self.gold_outputs[idx]
        
        # Encode input text
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze(),
            'gold_output': gold_output,
            'input_text': input_text
        }


def get_collate_fn(tokenizer, max_length):
    """Collate function for batching"""
    def collate_fn(batch):
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch])
        }
    return collate_fn


def get_collate_fn_for_gen_model(tokenizer, max_length):
    """Collate function for generation models"""
    def collate_fn(batch):
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch]),
            'gold_outputs': [item['gold_output'] for item in batch],
            'input_texts': [item['input_text'] for item in batch]
        }
    return collate_fn


def get_hf_tokenizer(model_name):
    """Get HuggingFace tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# Legacy SQuAD functions for backward compatibility
class SQuADDataset(Dataset):
    """
    SQuAD dataset for question answering with MeZO-style formatting
    Formats data as: "Context: ... Question: ... Answer: ..."
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
                'id': example['id']
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
            'answer': example['answer']
        }


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


def get_lm_dataloaders(task, tokenizer, train_batch_size=2, test_batch_size=2, 
                       max_length=512, num_train_examples=1000, num_eval_examples=200, seed=42):
    """
    Create dataloaders for various LM tasks (classification and generation)
    """
    
    if isinstance(task, str):
        if task in ["squad", "drop", "xsum"]:
            task = LmGenerationTask(task)
        elif task in ["sst2"]:
            task = LmClassificationTask(task)
        else:
            raise ValueError(f"Unknown task: {task}")
    
    # Load dataset
    if task.value == "squad":
        dataset = load_dataset(LM_DATASET_MAP[task.value])
        raw_train_dataset = dataset["train"].select(range(min(num_train_examples, len(dataset["train"])))).shuffle(seed)
        raw_test_dataset = dataset["validation"].select(range(min(num_eval_examples, len(dataset["validation"])))).shuffle(seed)
    elif task.value == "drop":
        dataset = load_dataset(LM_DATASET_MAP[task.value])
        raw_train_dataset = dataset["train"].select(range(min(num_train_examples, len(dataset["train"])))).shuffle(seed)
        raw_test_dataset = dataset["validation"].select(range(min(num_eval_examples, len(dataset["validation"])))).shuffle(seed)
    elif task.value == "xsum":
        dataset = load_dataset(LM_DATASET_MAP[task.value])
        raw_train_dataset = dataset["train"].select(range(min(num_train_examples, len(dataset["train"])))).shuffle(seed)
        raw_test_dataset = dataset["test"].select(range(min(num_eval_examples, len(dataset["test"])))).shuffle(seed)
    elif task.value == "sst2":
        dataset = load_dataset(LM_DATASET_MAP[task.value], task.value)
        raw_train_dataset = dataset["train"].select(range(min(num_train_examples, len(dataset["train"])))).shuffle(seed)
        raw_test_dataset = dataset["validation"].select(range(min(num_eval_examples, len(dataset["validation"])))).shuffle(seed)
    
    # Get template
    template = LM_TEMPLATE_MAP[task.value]()
    
    if isinstance(task, LmClassificationTask):
        # Classification task
        encoded_train_texts = list(map(template.verbalize, raw_train_dataset))
        encoded_test_texts = list(map(template.verbalize, raw_test_dataset))
        train_dataset = CustomLMDataset(encoded_train_texts, tokenizer, max_length=max_length)
        test_dataset = CustomLMDataset(encoded_test_texts, tokenizer, max_length=max_length)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=get_collate_fn(tokenizer, max_length)
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            collate_fn=get_collate_fn(tokenizer, max_length)
        )
        
    elif isinstance(task, LmGenerationTask):
        # Generation task
        encoded_train_texts = list(map(template.verbalize, raw_train_dataset))
        encoded_test_texts = list(map(template.encode, raw_test_dataset))
        
        # Extract gold outputs
        if task == LmGenerationTask.squad:
            test_golds = list(map(lambda d: d["answers"]["text"][0], raw_test_dataset))
        elif task == LmGenerationTask.drop:
            test_golds = list(map(lambda d: d["answers_spans"]["spans"][0], raw_test_dataset))
        elif task == LmGenerationTask.xsum:
            test_golds = list(map(lambda d: d["summary"], raw_test_dataset))
        
        train_dataset = CustomLMDataset(encoded_train_texts, tokenizer, max_length=max_length)
        test_dataset = CustomLMGenerationDataset(encoded_test_texts, test_golds, tokenizer, max_length=max_length)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=get_collate_fn(tokenizer, max_length)
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            collate_fn=get_collate_fn_for_gen_model(tokenizer, max_length)
        )
    
    return train_loader, test_loader


def get_squad_dataloaders(tokenizer, train_batch_size=2, test_batch_size=2, 
                         max_length=512, num_train_examples=1000, num_eval_examples=200):
    """
    Legacy function for backward compatibility
    """
    return get_lm_dataloaders(
        LmGenerationTask.squad, tokenizer, train_batch_size, test_batch_size,
        max_length, num_train_examples, num_eval_examples
    )


def extract_answer_from_generation(generated_text, question=None):
    """
    Extract answer from generated text for evaluation
    Looks for text after "Answer:" or "Summary:" in the generation
    """
    # Different patterns for different tasks
    patterns = ["Answer:", "Summary:", "Sentiment:"]
    
    for pattern in patterns:
        if pattern in generated_text:
            answer_part = generated_text.split(pattern)[-1].strip()
            # Take only the first sentence/line as the answer
            answer = answer_part.split('\n')[0].split('.')[0].strip()
            return answer
    
    # If no pattern found, return the generated text as is
    return generated_text.strip()


def calculate_generation_metrics(model, tokenizer, batch, device, max_new_tokens=50):
    """
    Calculate generation-specific metrics (F1, Exact Match, ROUGE for summarization)
    Also calculates accuracy from the forward pass
    """
    model.eval()
    
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    gold_outputs = batch.get('gold_outputs', batch.get('answers', []))
    
    f1_scores = []
    em_scores = []
    
    with torch.no_grad():
        # Forward pass for accuracy calculation
        forward_outputs = model(input_ids, attention_mask, labels)
        
        # Calculate accuracy from forward pass
        logits = forward_outputs.logits
        if logits.shape[1] != labels.shape[1]:
            min_len = min(logits.shape[1], labels.shape[1])
            logits = logits[:, :min_len, :]
            labels_aligned = labels[:, :min_len]
        else:
            labels_aligned = labels
        
        # For next token prediction: predict token i+1 from tokens 0:i
        if logits.shape[1] > 1:
            pred_logits = logits[:, :-1, :].contiguous()
            target_labels = labels_aligned[:, 1:].contiguous()
        else:
            pred_logits = logits.contiguous()
            target_labels = labels_aligned.contiguous()
        
        predictions = torch.argmax(pred_logits, dim=-1)
        mask = (target_labels != -100)
        
        if mask.sum() > 0:
            if predictions.shape != target_labels.shape:
                min_len = min(predictions.shape[1], target_labels.shape[1])
                predictions = predictions[:, :min_len]
                target_labels = target_labels[:, :min_len]
                mask = mask[:, :min_len]
            
            correct_predictions = (predictions == target_labels)[mask]
            accuracy = correct_predictions.float().mean().item()
        else:
            accuracy = 0.0
        
        # Generate outputs for F1/EM calculation
        if len(gold_outputs) > 0:  # Only if we have gold outputs
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode generated outputs
            for i, output in enumerate(outputs):
                if i < len(gold_outputs):
                    generated_text = tokenizer.decode(output, skip_special_tokens=True)
                    predicted_answer = extract_answer_from_generation(generated_text)
                    ground_truth = gold_outputs[i]
                    
                    # Calculate metrics
                    f1 = f1_score(predicted_answer, ground_truth)
                    em = exact_match_score(predicted_answer, ground_truth)
                    
                    f1_scores.append(f1)
                    em_scores.append(em)
    
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    
    return avg_f1, avg_em, accuracy


# Legacy function for backward compatibility - now returns accuracy too
def calculate_squad_metrics(model, tokenizer, batch, device, max_new_tokens=50):
    """Legacy function for SQuAD metrics - now includes accuracy"""
    return calculate_generation_metrics(model, tokenizer, batch, device, max_new_tokens)


def create_squad_training_setup(model_name='gpt2', num_train_examples=1000, num_eval_examples=200):
    """Legacy function for backward compatibility"""
    tokenizer = get_hf_tokenizer(model_name)
    train_loader, test_loader = get_lm_dataloaders(
        LmGenerationTask.squad, tokenizer, 
        train_batch_size=2, test_batch_size=2, max_length=512,
        num_train_examples=num_train_examples, num_eval_examples=num_eval_examples
    )
    return train_loader, test_loader, tokenizer


if __name__ == "__main__":
    # Test the enhanced dataset loading
    print("Testing enhanced LM dataset loading...")
    
    tokenizer = get_hf_tokenizer('gpt2')
    
    # Test SQuAD generation task
    print("\n=== Testing SQuAD Generation Task ===")
    train_loader, test_loader = get_lm_dataloaders(
        LmGenerationTask.squad, tokenizer,
        train_batch_size=2, test_batch_size=2, max_length=128,
        num_train_examples=10, num_eval_examples=5
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    
    # Print sample from training set
    for batch in train_loader:
        print("\nTraining batch sample:")
        print(f"Input shape: {batch['input_ids'].shape}")
        print(f"Sample text: {tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)[:200]}...")
        break
    
    # Print sample from test set (generation)
    for batch in test_loader:
        print("\nTest batch sample (generation):")
        print(f"Input shape: {batch['input_ids'].shape}")
        print(f"Gold outputs: {batch['gold_outputs'][0]}")
        print(f"Input text: {batch['input_texts'][0][:100]}...")
        break
    
    # Test other tasks
    print("\n=== Testing DROP Generation Task ===")
    try:
        train_loader, test_loader = get_lm_dataloaders(
            LmGenerationTask.drop, tokenizer,
            train_batch_size=1, test_batch_size=1, max_length=128,
            num_train_examples=2, num_eval_examples=2
        )
        print(f"DROP - Train: {len(train_loader)}, Test: {len(test_loader)}")
    except Exception as e:
        print(f"DROP loading failed: {e}")
    
    print("\n=== Testing XSum Generation Task ===")
    try:
        train_loader, test_loader = get_lm_dataloaders(
            LmGenerationTask.xsum, tokenizer,
            train_batch_size=1, test_batch_size=1, max_length=128,
            num_train_examples=2, num_eval_examples=2
        )
        print(f"XSum - Train: {len(train_loader)}, Test: {len(test_loader)}")
    except Exception as e:
        print(f"XSum loading failed: {e}")