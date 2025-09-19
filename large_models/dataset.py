import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import json
import re
from collections import Counter
import string
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

_all__ = [
    "build_task_datasets",  # main entry
    "TaskSpec",
]

@dataclass(frozen=True)
class TaskSpec:
    name: str
    hf_path: str
    config: Optional[str]
    split_train: str
    split_eval: str
    task_type: str  # "qa" | "cls" | "gen"
    formatter: Callable[[dict], Tuple[str, str, dict]]

def _format_squad(ex):
    # HF 'squad' and 'squad_v2' compatible
    ctx = ex["context"].strip()
    q = ex["question"].strip()
    # answers is dict: {"text": [...], "answer_start": [...]}
    refs = [t.strip() for t in ex.get("answers", {}).get("text", []) if t and t.strip()]
    tgt = refs[0] if refs else ""     # single target for LM loss; refs kept for metrics
    prompt = f"Context: {ctx}\nQuestion: {q}\nAnswer:"
    return prompt, tgt, {"refs": refs}

def _format_sst2(ex):
    sent = ex["sentence"].strip()
    label = int(ex["label"])  # 1=positive, 0=negative
    tgt = "positive" if label == 1 else "negative"
    prompt = f"Sentence: {sent}\nLabel (positive/negative):"
    return prompt, tgt, {"label_id": label}

def _format_drop(ex):
    # DROP (ucinlp/drop) has textual and numeric answers; normalize to list[str]
    passage = ex["passage"].strip()
    question = ex["question"].strip()
    # Prefer spans if present; fall back to 'answers' text if available
    spans = ex.get("answers_spans", {})
    text_spans = []
    for k in ("spans", "answer_spans", "texts"):
        v = spans.get(k, [])
        if isinstance(v, list):
            text_spans.extend([s.strip() for s in v if isinstance(s, str)])
    # numeric answers (e.g., counts)
    num = ex.get("number", None)
    if isinstance(num, (int, float)) and str(num) not in text_spans:
        text_spans.append(str(num))
    refs = [t for t in text_spans if t]
    tgt = refs[0] if refs else ""  # use a single canonical target for LM loss
    prompt = f"Context: {passage}\nQuestion: {question}\nAnswer:"
    return prompt, tgt, {"refs": refs}

def _format_xsum(ex):
    doc = ex["document"].strip()
    summ = ex["summary"].strip()
    prompt = f"Document:\n{doc}\n\nSummary:"
    return prompt, summ, {}

TASKS: Dict[str, TaskSpec] = {
    # SQuAD v1/v2: choose your path; v1 shown here
    "squad": TaskSpec("squad", "squad", None, "train", "validation", "qa", _format_squad),
    # GLUE/SST-2 (via nyu-mll/glue with config='sst2')
    "sst2": TaskSpec("sst2", "glue", "sst2", "train", "validation", "cls", _format_sst2),
    # DROP
    "drop": TaskSpec("drop", "ucinlp/drop", None, "train", "validation", "qa", _format_drop),
    # XSum
    "xsum": TaskSpec("xsum", "GEM/xsum", None, "train", "validation", "gen", _format_xsum),
}

def _tokenize_example(tokenizer: PreTrainedTokenizerBase, prompt: str, target: str, max_length: int):
    # Teacher-forced target appended after prompt for CausalLM; labels mask the prompt tokens.
    text = prompt + " " + target
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )
    ids = enc["input_ids"][0]
    attn = enc["attention_mask"][0]
    # Build labels: ignore prompt tokens
    with tokenizer.as_target_tokenizer():
        tgt_ids = tokenizer(target, truncation=True, max_length=max_length, padding=False, return_tensors="pt")["input_ids"][0]
    labels = torch.full_like(ids, -100)
    labels[-len(tgt_ids):] = tgt_ids  # only supervise the target suffix
    return ids, attn, labels

def _collate_lm(tokenizer: PreTrainedTokenizerBase):
    pad = tokenizer.pad_token_id
    def pad_stack(tensors, pad_value):
        max_len = max(t.size(0) for t in tensors)
        out = []
        for t in tensors:
            if t.size(0) < max_len:
                pad_t = torch.full((max_len - t.size(0),), pad_value, dtype=t.dtype)
                t = torch.cat([t, pad_t], dim=0)
            out.append(t)
        return torch.stack(out, dim=0)

    def fn(batch):
        input_ids = pad_stack([b["input_ids"] for b in batch], pad)
        attention_mask = pad_stack([b["attention_mask"] for b in batch], 0)
        labels = pad_stack([b["labels"] for b in batch], -100)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            # Keep human-readable for debugging & metrics:
            "prompt_text": [b["prompt_text"] for b in batch],
            "text_target": [b["text_target"] for b in batch],
            "meta": [b.get("meta", {}) for b in batch],
        }
    return fn

def _collate_cls():
    def fn(batch):
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=0),
            "attention_mask": torch.nn.utils.rnn.pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0),
            "labels": torch.tensor([b["meta"]["label_id"] for b in batch], dtype=torch.long),
            "prompt_text": [b["prompt_text"] for b in batch],
            "text_target": [b["text_target"] for b in batch],
            "meta": [b.get("meta", {}) for b in batch],
        }
    return fn

def _tokenize_pair(tokenizer: PreTrainedTokenizerBase, prompt: str, target: Optional[str], max_len: int):
    # For encoder-decoder use labels; for causal LM we create labels by masking the prompt part.
    enc = tokenizer(prompt, truncation=True, max_length=max_len, padding=False)
    if target is None:
        return enc["input_ids"], enc["attention_mask"], None

    # For causal LM training-on-target: append target and mask prompt tokens with -100
    with_target = tokenizer(prompt + " " + target, truncation=True, max_length=max_len, padding=False)
    n_prompt = len(enc["input_ids"])
    labels = [-100]*n_prompt + with_target["input_ids"][n_prompt:]
    return with_target["input_ids"], with_target["attention_mask"], labels

class _HFDataset(Dataset):
    def __init__(self, hf_ds, tokenizer, fmt_fn, max_len: int, task_type: str):
        self.hf_ds = hf_ds
        self.tok = tokenizer
        self.fmt = fmt_fn
        self.max_len = max_len
        self.task_type = task_type

    def __len__(self): return len(self.hf_ds)

    def __getitem__(self, idx):
        ex = self.hf_ds[idx]
        prompt, target, meta = self.fmt(ex)

        # Classification uses label_id supervised targets rather than text loss
        if self.task_type == "cls":
            input_ids, attn, _ = _tokenize_pair(self.tok, prompt, None, self.max_len)
            label_id = meta["label_id"]
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "label_id": torch.tensor(label_id, dtype=torch.long),
                "text_target": target,  # for pretty prints
            }

        # Generative (qa/sum) uses text targets (prompt+answer) with prompt masked by -100
        input_ids, attn, labels = _tokenize_pair(self.tok, prompt, target, self.max_len)
        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long) if labels is not None else None,
            "refs": meta.get("refs", []),
            "target_text": target,
            "prompt_text": prompt,
        }
        return item
    
def _pad_seq(batch, pad_id: int):
    max_len = max(len(b["input_ids"]) for b in batch)
    for b in batch:
        pad = max_len - len(b["input_ids"])
        if pad > 0:
            b["attention_mask"] = torch.tensor(
                list(b["attention_mask"]) + [0]*pad, dtype=torch.long
            )
            b["input_ids"] = torch.tensor(
                list(b["input_ids"]) + [pad_id]*pad, dtype=torch.long
            )
            if "labels" in b and b["labels"] is not None:
                b["labels"] = torch.tensor(
                    list(b["labels"]) + [-100]*pad, dtype=torch.long
                )
    return batch

def _collate_gen(tokenizer):
    pad_id = tokenizer.pad_token_id
    def fn(batch):
        batch = _pad_seq(batch, pad_id)
        out = {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        }
        if "labels" in batch[0] and batch[0]["labels"] is not None:
            out["labels"] = torch.stack([b["labels"] for b in batch])
        # carry strings for metrics
        out["refs"] = [b.get("refs", []) for b in batch]
        out["target_text"] = [b.get("target_text", "") for b in batch]
        out["prompt_text"] = [b.get("prompt_text", "") for b in batch]
        return out
    return fn

# def _collate_cls(tokenizer):
#     pad_id = tokenizer.pad_token_id
#     def fn(batch):
#         batch = _pad_seq(batch, pad_id)
#         return {
#             "input_ids": torch.stack([b["input_ids"] for b in batch]),
#             "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
#             "labels": torch.stack([b["label_id"] for b in batch]),
#             "target_text": [b.get("text_target","") for b in batch],
#         }
#     return fn

def _spec_squad():
    return TaskSpec(
        name="squad",
        task_type="qa",
        hf_path="squad",
        hf_config=None,
        splits={"train":"train", "validation":"validation", "test":"validation"},
        format_fn=_format_squad,
        collate_fn=None,  # filled later (needs tokenizer)
    )

def _spec_sst2():
    return TaskSpec(
        name="sst2",
        task_type="cls",
        hf_path="glue",
        hf_config="sst2",
        splits={"train":"train", "validation":"validation", "test":"validation"},
        format_fn=_format_sst2,
        collate_fn=None,
    )

def _spec_drop():
    return TaskSpec(
        name="drop",
        task_type="qa",
        hf_path="drop",
        hf_config=None,
        splits={"train":"train", "validation":"validation", "test":"validation"},
        format_fn=_format_drop,
        collate_fn=None,
    )

def _spec_xsum():
    return TaskSpec(
        name="xsum",
        task_type="sum",
        hf_path="xsum",
        hf_config=None,
        splits={"train":"train", "validation":"validation", "test":"validation"},
        format_fn=_format_xsum,
        collate_fn=None,
    )

_REGISTRY: Dict[str, Callable[[], TaskSpec]] = {
    "squad": _spec_squad,
    "sst2": _spec_sst2,
    "drop":  _spec_drop,
    "xsum":  _spec_xsum,
}

# def build_task_datasets(
#     task: str,
#     tokenizer: PreTrainedTokenizerBase,
#     max_length: int = 512,
#     streaming: bool = False,):
#     """
#     Returns: (spec, train_ds, val_ds, test_ds, collate_fn)
#     """
#     t = task.lower()
#     if t not in _REGISTRY:
#         raise ValueError(f"Unsupported task '{task}'. Choose from {list(_REGISTRY)}")

#     spec = _REGISTRY[t]()


#     # choose collator
#     # spec.collate_fn = _collate_cls(tokenizer) if spec.task_type == "cls" else _collate_gen(tokenizer)

#     # Load splits
#     def _ld(split):
#         return load_dataset(spec.hf_path, spec.hf_config, split=split, streaming=streaming)

#     raw_train = _ld(spec.splits["train"])
#     raw_val   = _ld(spec.splits["validation"])
#     raw_test  = _ld(spec.splits["test"])

#     train_ds = _HFDataset(raw_train, tokenizer, spec.format_fn, max_length, spec.task_type)
#     val_ds   = _HFDataset(raw_val,   tokenizer, spec.format_fn, max_length, spec.task_type)
#     test_ds  = _HFDataset(raw_test,  tokenizer, spec.format_fn, max_length, spec.task_type)

#     return spec, train_ds, val_ds, test_ds, spec.collate_fn


def build_task_datasets(task: str, tokenizer: PreTrainedTokenizerBase, max_length: int = 512,
                        num_train_examples: Optional[int] = None,
                        num_eval_examples: Optional[int] = None):
    task = task.lower()
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Supported: {list(TASKS.keys())}")
    spec = TASKS[task]

    # Load HF dataset
    if spec.config:
        ds = load_dataset(spec.hf_path, spec.config)
    else:
        ds = load_dataset(spec.hf_path)

    train_raw = ds[spec.split_train]
    eval_raw = ds[spec.split_eval]

    def _prep(example):
        prompt, tgt, meta = spec.formatter(example)
        ids, attn, labels = _tokenize_example(tokenizer, prompt, tgt, max_length)
        return {
            "input_ids": ids,
            "attention_mask": attn,
            "labels": labels if spec.task_type != "cls" else torch.zeros_like(ids),  # placeholder for uniformity
            "prompt_text": prompt,
            "text_target": tgt,
            "meta": meta,
        }

    train = [ _prep(ex) for ex in (train_raw if num_train_examples is None else train_raw.select(range(num_train_examples))) ]
    evald = [ _prep(ex) for ex in (eval_raw if num_eval_examples is None else eval_raw.select(range(num_eval_examples))) ]

    collate = _collate_cls() if spec.task_type == "cls" else _collate_lm(tokenizer)

    return train, evald, collate, spec.task_type

from torch.utils.data import DataLoader

def get_dataloaders(task, tokenizer, train_bs=8, eval_bs=8, max_length=512,
                    num_train_examples=None, num_eval_examples=None, shuffle=True):
    train, evald, collate, task_type = build_task_datasets(
        task, tokenizer, max_length, num_train_examples, num_eval_examples
    )
    train_loader = DataLoader(train, batch_size=train_bs, shuffle=shuffle, collate_fn=collate)
    eval_loader  = DataLoader(evald, batch_size=eval_bs,  shuffle=False,   collate_fn=collate)
    return train_loader, eval_loader, task_type

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
                dataset = load_dataset('EdinburghNLP/xsum', split=split)
            except:
                try:
                    # Try alternative XSum loading
                    dataset = load_dataset('xsum', split=split)
                except:
                    # Last resort: try CNN/DailyMail
                    alt_split = 'validation' if split == 'validation' else ('train' if split == 'train' else 'test')
                    dataset = load_dataset('cnn_dailymail', '3.0.0', split=alt_split)
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
            
            if not answer and 'number' in example:
                answer = str(example['number'])
            
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
            
        elif task == "sst2":
            # Robustly read fields
            sentence = example.get("sentence", "")
            raw_label = example.get("label", 0)
            try:
                label = int(raw_label)
            except Exception:
                # Some loaders yield strings like "1"
                label = 1 if str(raw_label).strip() in {"1", "true", "pos", "positive"} else 0

            # Verbalizer must stay consistent across train/eval to keep EM/F1 meaningful
            verbalizer = {0: "terrible", 1: "great"}
            completion_word = verbalizer[label]

            # Use the same universal prompt shape your SQuAD path uses
            question = "What is the overall sentiment of the sentence?"
            formatted_text = (
                f"Context: {sentence}\n"
                f"Question: {question}\n"
                f"Answer: {completion_word}"
            )

            # Return keys that your collate/eval already understand
            return {
                "text": formatted_text,                # legacy key (if used elsewhere)
                "context": sentence,
                "question": question,
                "answer": completion_word,             # keep if some code expects 'answer'
                "answer_text": completion_word,        # common in your QA path
                "answers": [completion_word],          # safe default for EM/F1 helpers
                "id": example.get("idx", example.get("guid", "")),
                "formatted_text": formatted_text,      # the string the model actually sees
                "original_example": example,
                "label": label,                        # keep raw label for debugging
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