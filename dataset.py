import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import re
from collections import Counter
import string

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
            # Ensure the suffix containing "Answer:" + answer is retained
            old_side = getattr(self.tokenizer, 'truncation_side', 'right')
            try:
                self.tokenizer.truncation_side = 'left'
            except Exception:
                pass
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            try:
                self.tokenizer.truncation_side = old_side
            except Exception:
                pass
            
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            
            # Ensure consistent tensor shapes
            if input_ids.dim() == 0:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask.dim() == 0:
                attention_mask = attention_mask.unsqueeze(0)
            
            # Build labels that supervise ONLY the answer tokens (ignore prefix + padding)
            labels = input_ids.clone()
            valid_len = int(attention_mask.sum().item())
            ids_list = input_ids.tolist()

            # Helper to find token subsequence
            def _find_subseq(hay, needle):
                n = len(needle)
                if n == 0 or n > len(hay):
                    return -1
                for i in range(0, len(hay) - n + 1):
                    if hay[i:i+n] == needle:
                        return i
                return -1

            # Try several encodings of the marker to be robust to whitespace/newlines
            candidates = []
            for marker in ["\nAnswer:", " Answer:", "Answer:"]:
                try:
                    tok = self.tokenizer.encode(marker, add_special_tokens=False)
                    if tok:
                        candidates.append(tok)
                except Exception:
                    pass

            pos = -1
            match_len = 0
            for cand in candidates:
                p = _find_subseq(ids_list, cand)
                if p != -1 and (pos == -1 or p < pos):
                    pos = p
                    match_len = len(cand)

            if pos != -1:
                start_idx = pos + match_len
            else:
                # Fallback via character split
                astart = text.find("Answer:")
                if astart == -1:
                    start_idx = 0
                else:
                    prefix = text[:astart + len("Answer:")]
                    old_side2 = getattr(self.tokenizer, 'truncation_side', 'right')
                    try:
                        self.tokenizer.truncation_side = 'left'
                    except Exception:
                        pass
                    prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
                    try:
                        self.tokenizer.truncation_side = old_side2
                    except Exception:
                        pass
                    start_idx = len(prefix_tokens)

            # Clamp to valid sequence length
            start_idx = max(0, min(start_idx, valid_len))
            # Ignore everything before the answer prefix
            if start_idx > 0:
                labels[:start_idx] = -100
            # Ignore padding
            labels[attention_mask == 0] = -100
            pad_id = getattr(self.tokenizer, 'pad_token_id', None)
            if pad_id is not None:
                labels[input_ids == pad_id] = -100
            labels = labels.long()
            
            result = {
                'input_ids': input_ids,
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
        elif task == 'boolq':
            dataset = load_dataset('boolq', split=split)
        elif task == 'copa':
            dataset = load_dataset('super_glue', 'copa', split=split)
        elif task == 'multirc':
            dataset = load_dataset('super_glue', 'multirc', split=split)
        elif task == 'cb':
            dataset = load_dataset('super_glue', 'cb', split=split)
        elif task == 'wic':
            dataset = load_dataset('super_glue', 'wic', split=split)
        elif task == 'wsc':
            dataset = load_dataset('super_glue', 'wsc.fixed', split=split)
        elif task == 'record':
            dataset = load_dataset('super_glue', 'record', split=split)
        elif task == 'rte':
            dataset = load_dataset('super_glue', 'rte', split=split)
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

            # Use the same prompt shape as QA so metrics can locate Answer tokens
            question = "Write a concise, faithful summary of the document."
            formatted_text = (
                f"Context: {document}\n"
                f"Question: {question}\n"
                f"Answer: {summary}"
            )

            return {
                'text': formatted_text,
                'context': document,
                'question': question,
                'answer': summary,
                'answers': [summary],
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

        elif task == 'boolq':
            passage = example.get('passage', '')
            question = example.get('question', '')
            answer_bool = bool(example.get('answer', False))
            answer = 'Yes' if answer_bool else 'No'
            formatted_text = f"Context: {passage}\nQuestion: {question}\nAnswer: {answer}"
            return {
                'text': formatted_text,
                'context': passage,
                'question': question,
                'answer': answer,
                'answers': [answer],
                'id': example.get('id', example.get('idx', '')),
                'formatted_text': formatted_text,
                'original_example': example
            }

        elif task == 'copa':
            premise = example.get('premise', '')
            qtype = example.get('question', '')  # 'cause' or 'effect'
            choice1 = example.get('choice1', '')
            choice2 = example.get('choice2', '')
            label = int(example.get('label', 0))
            choices_text = f"(A) {choice1} (B) {choice2}"
            ask = 'cause' if qtype == 'cause' else 'effect'
            answer = choice1 if label == 0 else choice2
            formatted_text = (
                f"Context: {premise}\nQuestion: Choose the most plausible {ask}. {choices_text}\nAnswer: {answer}"
            )
            return {
                'text': formatted_text,
                'context': premise,
                'question': f"Choose the most plausible {ask}.",
                'answer': answer,
                'answers': [answer],
                'id': example.get('idx', example.get('id', '')),
                'formatted_text': formatted_text,
                'original_example': example
            }

        elif task == 'multirc':
            paragraph = example.get('paragraph', '')
            question = example.get('question', '')
            answer_text = example.get('answer', '')
            label = int(example.get('label', 0))
            yn = 'Yes' if label == 1 else 'No'
            formatted_text = (
                f"Context: {paragraph}\nQuestion: {question}\nI found this answer \"{answer_text}\". Is that correct?\nAnswer: {yn}"
            )
            return {
                'text': formatted_text,
                'context': paragraph,
                'question': question,
                'answer': yn,
                'answers': [yn],
                'id': example.get('idx', example.get('id', '')),
                'formatted_text': formatted_text,
                'original_example': example
            }

        elif task == 'cb':
            premise = example.get('premise', '')
            hypothesis = example.get('hypothesis', '')
            label = int(example.get('label', 0))  # 0,1,2
            mapping = {0: 'Yes', 1: 'No', 2: 'Maybe'}
            answer = mapping.get(label, 'Maybe')
            formatted_text = (
                f"Context: {premise}\nQuestion: Can we infer that \"{hypothesis}\"? Yes, No, or Maybe?\nAnswer: {answer}"
            )
            return {
                'text': formatted_text,
                'context': premise,
                'question': f"Can we infer that \"{hypothesis}\"?",
                'answer': answer,
                'answers': [answer],
                'id': example.get('idx', example.get('id', '')),
                'formatted_text': formatted_text,
                'original_example': example
            }

        elif task == 'wic':
            sent1 = example.get('sentence1', '')
            sent2 = example.get('sentence2', '')
            word = example.get('word', '')
            label = int(example.get('label', 0))
            answer = 'Yes' if label == 1 else 'No'
            formatted_text = (
                f"Context: Does the word \"{word}\" have the same meaning in these two sentences?\n"
                f"{sent1}\n{sent2}\nAnswer: {answer}"
            )
            return {
                'text': formatted_text,
                'context': f"{sent1}\n{sent2}",
                'question': f"Does the word \"{word}\" have the same meaning?",
                'answer': answer,
                'answers': [answer],
                'id': example.get('idx', example.get('id', '')),
                'formatted_text': formatted_text,
                'original_example': example
            }

        elif task == 'wsc':
            text = example.get('text', '')
            span1 = example.get('span1_text', '')
            span2 = example.get('span2_text', '')
            label = int(example.get('label', 0))
            answer = 'Yes' if label == 1 else 'No'
            formatted_text = (
                f"Context: {text}\nQuestion: Does the pronoun \"{span2}\" refer to {span1}?\nAnswer: {answer}"
            )
            return {
                'text': formatted_text,
                'context': text,
                'question': f"Does the pronoun \"{span2}\" refer to {span1}?",
                'answer': answer,
                'answers': [answer],
                'id': example.get('idx', example.get('id', '')),
                'formatted_text': formatted_text,
                'original_example': example
            }

        elif task == 'record':
            passage = example.get('passage', '')
            query = example.get('query', '')
            answers_list = example.get('answers', [])
            answer = answers_list[0] if isinstance(answers_list, list) and len(answers_list) > 0 else ''
            formatted_text = f"Context: {passage}\nQuestion: {query}\nAnswer: {answer}"
            return {
                'text': formatted_text,
                'context': passage,
                'question': query,
                'answer': answer,
                'answers': list(answers_list) if isinstance(answers_list, list) else ([answer] if answer else []),
                'id': example.get('idx', example.get('id', '')),
                'formatted_text': formatted_text,
                'original_example': example
            }

        elif task == 'rte':
            premise = example.get('premise', '')
            hypothesis = example.get('hypothesis', '')
            label = int(example.get('label', 0))
            answer = 'Yes' if label == 0 else 'No'
            formatted_text = (
                f"Context: {premise}\nQuestion: Does this mean that \"{hypothesis}\" is true?\nAnswer: {answer}"
            )
            return {
                'text': formatted_text,
                'context': premise,
                'question': f"Is \"{hypothesis}\" entailed?",
                'answer': answer,
                'answers': [answer],
                'id': example.get('idx', example.get('id', '')),
                'formatted_text': formatted_text,
                'original_example': example
            }

    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example['text']
        
        # Tokenize like SQuAD but ensure the suffix (answers) stay supervised
        old_side = getattr(self.tokenizer, 'truncation_side', 'right')
        try:
            self.tokenizer.truncation_side = 'left'
        except Exception:
            pass
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        try:
            self.tokenizer.truncation_side = old_side
        except Exception:
            pass
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()

        valid_len = int(attention_mask.sum().item())
        ids_list = input_ids.tolist()

        def _find_subseq(hay, needle):
            n = len(needle)
            if n == 0 or n > len(hay):
                return -1
            for i in range(0, len(hay) - n + 1):
                if hay[i:i+n] == needle:
                    return i
            return -1

        candidates = []
        for marker in ["\nAnswer:", " Answer:", "Answer:"]:
            try:
                tok = self.tokenizer.encode(marker, add_special_tokens=False)
                if tok:
                    candidates.append(tok)
            except Exception:
                pass

        pos = -1
        match_len = 0
        for cand in candidates:
            p = _find_subseq(ids_list, cand)
            if p != -1 and (pos == -1 or p < pos):
                pos = p
                match_len = len(cand)

        if pos != -1:
            start_idx = pos + match_len
        else:
            astart = text.find("Answer:")
            if astart == -1:
                start_idx = 0
            else:
                prefix = text[:astart + len("Answer:")]
                old_side2 = getattr(self.tokenizer, 'truncation_side', 'right')
                try:
                    self.tokenizer.truncation_side = 'left'
                except Exception:
                    pass
                prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
                try:
                    self.tokenizer.truncation_side = old_side2
                except Exception:
                    pass
                start_idx = len(prefix_tokens)

        start_idx = max(0, min(start_idx, valid_len))
        if start_idx > 0:
            labels[:start_idx] = -100
        labels[attention_mask == 0] = -100
        pad_id = getattr(self.tokenizer, 'pad_token_id', None)
        if pad_id is not None:
            labels[input_ids == pad_id] = -100
        labels = labels.long()

        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'example_id': example['id'],
            'answer': example['answer'],
            'formatted_text': example['formatted_text'],
            'original_example': example['original_example']
        }

        # Provide a discrete class label for classification tasks like SST-2
        if self.task == 'sst2':
            # Prefer explicit label if present; else infer from answer word
            label = example.get('label', None)
            if label is None:
                ans = str(example.get('answer', '')).strip().lower()
                label = 1 if ans == 'great' else 0
            item['class_label'] = int(label)

        return item

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
            # Optional fields (some codepaths expect server_kv_state even in LoRA mode)
            formatted_texts.append(item.get('formatted_text', ""))
            original_examples.append(item.get('original_example', {}))
            if 'server_kv_state' in item:
                pass
        
        # Stack tensors
        batch_dict = {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks),
            'labels': torch.stack(labels),
        }
        
        # Always include non-tensor metadata lists for downstream logging/metrics
        batch_dict['formatted_text'] = formatted_texts
        batch_dict['original_example'] = original_examples
        
        return batch_dict
        
    except Exception as e:
        print(f"Collate function error: {e}")
        print(f"Batch info: {len(batch)} items")
        for i, item in enumerate(batch):
            print(f"  Item {i}: keys={list(item.keys())}, shapes={[item[k].shape if torch.is_tensor(item[k]) else type(item[k]) for k in item.keys()]}")
        raise


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
        class_labels = []
        answers_flat = []
        single_answers = []
        
        for item in batch:
            input_ids.append(item['input_ids'])
            attention_masks.append(item['attention_mask'])
            labels.append(item['labels'])
            formatted_texts.append(item['formatted_text'])
            original_examples.append(item['original_example'])
            # Optional: include discrete class labels when available (e.g., SST-2)
            if 'class_label' in item:
                class_labels.append(int(item['class_label']))
            # Collect answers for metrics
            if 'answers' in item and isinstance(item['answers'], list):
                answers_flat.append(item['answers'])
            else:
                answers_flat.append([item.get('answer', '')])
            single_answers.append(item.get('answer', ''))
        
        batch_out = {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks),
            'labels': torch.stack(labels),
            'formatted_text': formatted_texts,
            'original_example': original_examples,
            'answers': answers_flat,
            'answer': single_answers
        }
        if len(class_labels) == len(input_ids) and len(class_labels) > 0:
            batch_out['class_label'] = torch.tensor(class_labels, dtype=torch.long)
        return batch_out
    
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