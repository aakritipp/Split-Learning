"""
FIXED: Multi-task metrics calculation module
Handles SQUAD, SST2, DROP, XSum, and other tasks with proper ground truth extraction
"""

import torch
import numpy as np
import re
import string
from collections import Counter
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

try:
    import evaluate  # pip install evaluate
    _has_evaluate = True
except Exception:
    _has_evaluate = False

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
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

def _qa_em_f1(pred: str, golds: list):
    pred_n = normalize_answer(pred)
    gold_ns = [normalize_answer(g) for g in golds if g is not None]
    if not gold_ns:
        return 0.0, 0.0
    em = float(any(pred_n == g for g in gold_ns))
    def f1(p, g):
        ps, gs = p.split(), g.split()
        common = Counter(ps) & Counter(gs)
        num_same = sum(common.values())
        if num_same == 0: return 0.0
        prec = num_same / max(len(ps), 1)
        rec  = num_same / max(len(gs), 1)
        return 2 * prec * rec / (prec + rec)
    f1s = [f1(pred_n, g) for g in gold_ns]
    return em, float(max(f1s))

def _cls_map_prediction_to_label(text_pred: str):
    s = (text_pred or "").lower()
    if "positive" in s: return 1
    if "negative" in s: return 0
    # numeric fallback
    if s.strip() in {"1", "pos", "+1"}: return 1
    if s.strip() in {"0", "neg", "-1"}: return 0
    # default: treat anything else as negative (conservative)
    return 0

def _rouge_scores(preds: list, refs: list):
    if _has_evaluate:
        rouge = evaluate.load("rouge")  # ROUGE-1/2/Lsum
        sc = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
        return {"rouge1": float(sc.get("rouge1", 0.0)),
                "rouge2": float(sc.get("rouge2", 0.0)),
                "rougeL": float(sc.get("rougeL", 0.0))}
    # Fallback: ROUGE-1 proxy
    def r1(p, r):
        p_set, r_set = set((p or "").split()), set((r or "").split())
        return len(p_set & r_set) / max(len(r_set), 1)
    return {"rouge1": float(np.mean([r1(p, r) for p, r in zip(preds, refs)])),
            "rouge2": 0.0, "rougeL": 0.0}

def compute_metrics_for_batch(task_type: str,
                              pred_texts: list,
                              batch_meta: list,
                              gold_texts: list):
    """
    task_type: "qa" | "cls" | "gen"
    pred_texts: decoded model outputs (len = batch)
    batch_meta: per-example dicts from dataset (contains 'refs' for QA, 'label_id' for CLS)
    gold_texts: the single canonical 'text_target' we trained against (len = batch)
    """
    if task_type == "qa":
        ems, f1s = [], []
        for pred, meta in zip(pred_texts, batch_meta):
            refs = meta.get("refs", [])
            em, f1 = _qa_em_f1(pred, refs if isinstance(refs, list) else [])
            ems.append(em); f1s.append(f1)
        return {"exact_match": float(np.mean(ems)), "f1": float(np.mean(f1s))}

    if task_type == "cls":
        gold = [m.get("label_id") for m in batch_meta]
        pred_ids = [_cls_map_prediction_to_label(p) for p in pred_texts]
        acc = float(np.mean([int(p == g) for p, g in zip(pred_ids, gold)]))
        return {"accuracy": acc}

    if task_type == "gen":
        # Use gold single-reference texts for ROUGE
        return _rouge_scores(pred_texts, gold_texts)

    return {}
    
def _f1_em(pred: str, refs: List[str]) -> Tuple[float,float]:
    # Token overlap F1 and exact match across any reference
    pred_norm = _normalize_text(pred)
    best_f1 = 0.0
    best_em = 0.0
    for r in refs or [""]:
        r_norm = _normalize_text(r)
        em = 1.0 if pred_norm == r_norm else 0.0
        pred_toks = pred_norm.split()
        ref_toks  = r_norm.split()
        common = {}
        for t in pred_toks:
            common[t] = min(pred_toks.count(t), ref_toks.count(t))
        num_same = sum(common.values())
        if len(pred_toks) == 0 and len(ref_toks) == 0:
            f1 = 1.0
        elif len(pred_toks) == 0 or len(ref_toks) == 0:
            f1 = 0.0
        else:
            precision = num_same / max(len(pred_toks), 1)
            recall    = num_same / max(len(ref_toks), 1)
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
        best_em = max(best_em, em)
    return best_f1, best_em

def _accuracy(pred_ids: List[int], label_ids: List[int]) -> float:
    pred_ids = np.asarray(pred_ids)
    label_ids = np.asarray(label_ids)
    return float((pred_ids == label_ids).mean())

def _rouge(preds: List[str], refs: List[str]) -> Dict[str,float]:
    if _has_evaluate:
        rouge = evaluate.load("rouge")
        sc = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
        # Ensure consistent keys
        return {
            "rouge1": float(sc.get("rouge1", 0.0)),
            "rouge2": float(sc.get("rouge2", 0.0)),
            "rougeL": float(sc.get("rougeL", 0.0)),
        }
    # Fallback: simplistic unigram overlap as ROUGE-1 proxy
    def _r1(p, r):
        p_set, r_set = set(p.split()), set(r.split())
        inter = len(p_set & r_set)
        return inter / max(len(r_set), 1)
    r1 = np.mean([_r1(p, r) for p, r in zip(preds, refs)])
    return {"rouge1": float(r1), "rouge2": 0.0, "rougeL": 0.0}

def compute_metrics_for_task(
    task_type: str,
    *,
    # For CLS:
    pred_label_ids: Optional[List[int]] = None,
    true_label_ids: Optional[List[int]] = None,
    # For QA/SUM:
    pred_texts: Optional[List[str]] = None,
    ref_texts_list: Optional[List[List[str]]] = None,  # list of refs per example
) -> Dict[str, float]:
    """
    Unifies metric calculation across tasks.
    task_type in {"cls", "qa", "sum"}.
    """
    if task_type == "cls":
        if pred_label_ids is None or true_label_ids is None:
            raise ValueError("Classification metrics need predicted and true label ids.")
        acc = _accuracy(pred_label_ids, true_label_ids)
        return {"accuracy": acc}

    if task_type == "qa":
        assert pred_texts is not None and ref_texts_list is not None
        f1s, ems = [], []
        for pred, refs in zip(pred_texts, ref_texts_list):
            f1, em = _f1_em(pred or "", refs or [])
            f1s.append(f1); ems.append(em)
        return {"f1": float(np.mean(f1s)), "em": float(np.mean(ems))}

    if task_type == "sum":
        assert pred_texts is not None and ref_texts_list is not None
        refs = [ (refs[0] if refs else "") for refs in ref_texts_list ]
        return _rouge(pred_texts, refs)

    raise ValueError(f"Unknown task_type: {task_type}")
    
       
def get_ground_truth_answers(batch, index):
    """
    FIXED: Multi-task ground truth extraction
    Handles SQUAD, SST2, DROP, XSum with robust fallback strategies
    """
    try:
        # Strategy 1: Check processed 'answers' field (your dataset creates this)
        if 'answers' in batch and index < len(batch['answers']):
            answers = batch['answers'][index]
            if isinstance(answers, list) and len(answers) > 0:
                valid_answers = [str(ans).strip() for ans in answers if str(ans).strip()]
                if valid_answers:
                    return valid_answers
        
        # Strategy 2: Check processed 'answer' field (single answer)
        if 'answer' in batch and index < len(batch['answer']):
            single_answer = batch['answer'][index]
            if isinstance(single_answer, str) and single_answer.strip():
                return [single_answer.strip()]
        
        # Strategy 3: Handle different original_example formats
        if 'original_example' in batch and index < len(batch['original_example']):
            original_ex = batch['original_example'][index]
            
            if isinstance(original_ex, dict):
                # SQUAD format: answers.text
                if ('answers' in original_ex and 
                    isinstance(original_ex['answers'], dict) and
                    'text' in original_ex['answers']):
                    squad_answers = original_ex['answers']['text']
                    if isinstance(squad_answers, list) and len(squad_answers) > 0:
                        return [str(ans) for ans in squad_answers if str(ans).strip()]
                
                # SST2 format: reconstruct from label
                if 'label' in original_ex:
                    try:
                        label = int(original_ex['label'])
                        if label in [0, 1]:
                            # CRITICAL: Must match your dataset.py verbalizer exactly
                            verbalizer = {0: "terrible", 1: "great"}  
                            return [verbalizer[label]]
                    except (ValueError, TypeError):
                        pass
                
                # DROP format: try various answer fields
                if 'answers_spans' in original_ex:
                    spans_data = original_ex['answers_spans']
                    if isinstance(spans_data, dict) and 'spans' in spans_data:
                        spans = spans_data['spans']
                        if isinstance(spans, list) and len(spans) > 0:
                            valid_spans = [str(span).strip() for span in spans if str(span).strip()]
                            if valid_spans:
                                return valid_spans
                
                # Direct answer field (some datasets)
                if 'answer' in original_ex:
                    direct_answer = str(original_ex['answer']).strip()
                    if direct_answer and direct_answer.lower() != 'none':
                        return [direct_answer]
        
        # Strategy 4: Extract from formatted_text as last resort
        if 'formatted_text' in batch and index < len(batch['formatted_text']):
            formatted_text = batch['formatted_text'][index]
            if "Answer:" in formatted_text:
                answer_part = formatted_text.split("Answer:")[-1].strip()
                if answer_part:
                    # Clean up the answer (remove newlines, periods)
                    answer_clean = answer_part.split('\n')[0].split('.')[0].strip()
                    if answer_clean:
                        return [answer_clean]
        
        return []  # No ground truth found
        
    except Exception as e:
        print(f"Error extracting ground truth for example {index}: {e}")
        return []


def squad_f1_score(prediction, ground_truth_list):
    """
    Calculate F1 score - handles multiple ground truth answers
    """
    try:
        # Handle the case where ground_truth_list is a single string
        if isinstance(ground_truth_list, str):
            ground_truth_list = [ground_truth_list]
        
        if not ground_truth_list or not prediction:
            return 0.0
        
        # Handle special cases like "CANNOTANSWER" or "no answer"
        if (len(ground_truth_list) > 0 and 
            (ground_truth_list[0] == "CANNOTANSWER" or ground_truth_list[0] == "no answer")):
            return float(normalize_answer(ground_truth_list[0]) == normalize_answer(prediction))
        
        # Calculate F1 for each possible answer and take maximum
        all_f1s = []
        for ground_truth in ground_truth_list:
            prediction_tokens = normalize_answer(prediction).split()
            ground_truth_tokens = normalize_answer(ground_truth).split()
            
            # Handle empty cases
            if len(prediction_tokens) == 0 and len(ground_truth_tokens) == 0:
                all_f1s.append(1.0)
                continue
            elif len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
                all_f1s.append(0.0)
                continue
            
            # Calculate token overlap
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            
            if num_same == 0:
                all_f1s.append(0.0)
            else:
                precision = 1.0 * num_same / len(prediction_tokens)
                recall = 1.0 * num_same / len(ground_truth_tokens)
                f1 = (2 * precision * recall) / (precision + recall)
                all_f1s.append(f1)
        
        # Return maximum F1 across all possible answers
        return float(max(all_f1s)) if all_f1s else 0.0
        
    except Exception as e:
        print(f"F1 calculation error: {e}")
        return 0.0


def squad_exact_match(prediction, ground_truth_list):
    """
    Calculate exact match - handles multiple ground truth answers
    """
    try:
        # Handle the case where ground_truth_list is a single string
        if isinstance(ground_truth_list, str):
            ground_truth_list = [ground_truth_list]
        
        if not ground_truth_list or not prediction:
            return 0.0
        
        # Check exact match against any of the ground truth answers
        normalized_prediction = normalize_answer(prediction)
        
        for ground_truth in ground_truth_list:
            normalized_ground_truth = normalize_answer(ground_truth)
            if normalized_prediction == normalized_ground_truth:
                return 1.0
        
        return 0.0
        
    except Exception as e:
        print(f"EM calculation error: {e}")
        return 0.0


def calculate_answer_token_accuracy(predictions, labels, batch, tokenizer):
    """Calculate accuracy only on answer portion tokens"""
    try:
        if 'formatted_text' not in batch or tokenizer is None:
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
        print(f"Answer token accuracy failed: {e}")
        return 0.0


def calculate_client_answer_accuracy(predictions, labels, batch, tokenizer):
    """Calculate accuracy only on answer portion tokens (client-side)"""
    try:
        if 'formatted_text' not in batch or tokenizer is None:
            # Fallback to general accuracy on valid tokens
            mask = (labels != -100) & (labels != 0)
            
            if mask.sum() == 0:
                # Try alternative: focus on non-prefix tokens
                prefix_len = 5  # Typical prefix length
                if labels.shape[1] > prefix_len:
                    start_idx = prefix_len
                    end_idx = min(labels.shape[1] - 1, start_idx + 50)  # Take next 50 tokens
                    
                    relevant_labels = labels[:, start_idx:end_idx]
                    relevant_preds = predictions[:, start_idx:end_idx]
                    mask = (relevant_labels != -100) & (relevant_labels != 0)
                    
                    if mask.sum() > 0:
                        correct = (relevant_preds == relevant_labels) & mask
                        return correct.sum().float() / mask.sum().float()
                    else:
                        return 0.0
                else:
                    return 0.0
            else:
                correct = (predictions == labels) & mask
                return correct.sum().float() / mask.sum().float()
        
        accuracies = []
        
        for i in range(len(batch['formatted_text'])):
            text = batch['formatted_text'][i]
            
            # Find "Answer:" position
            answer_start = text.find("Answer:")
            if answer_start == -1:
                continue
                
            # Get the answer part
            answer_text = text[answer_start + len("Answer:"):].strip()
            if not answer_text:
                continue
            
            # Tokenize to find answer token positions
            context_question = text[:answer_start + len("Answer:")]
            
            # Encode separately to find token boundaries
            context_tokens = tokenizer.encode(context_question, add_special_tokens=False)
            answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
            
            if len(answer_tokens) == 0:
                continue
            
            # Find answer token positions in the sequence
            start_idx = len(context_tokens)
            end_idx = min(start_idx + len(answer_tokens), predictions.shape[1])
            
            if start_idx < end_idx and start_idx < predictions.shape[1]:
                # Get predictions and labels for answer tokens
                answer_preds = predictions[i, start_idx:end_idx]
                answer_labels = labels[i, start_idx:end_idx]
                
                if len(answer_preds) > 0:
                    # Only count valid answer tokens (not -100)
                    valid_mask = (answer_labels != -100)
                    if valid_mask.sum() > 0:
                        correct = (answer_preds == answer_labels) & valid_mask
                        accuracy = correct.sum().float() / valid_mask.sum().float()
                        accuracies.append(accuracy.item())
        
        return sum(accuracies) / len(accuracies) if accuracies else 0.0
        
    except Exception as e:
        print(f"Client answer accuracy calculation failed: {e}")
        return 0.0



def calculate_generation_f1_em(model, batch, tokenizer, device, max_new_tokens=5):
    """
    FIXED: Generation with constrained decoding for classification tasks
    """
    try:
        model.eval()
        f1_scores = []
        em_scores = []
        
        print(f"\nGeneration Debug: Processing {len(batch.get('formatted_text', []))} examples")
        
        with torch.no_grad():
            for i in range(len(batch.get('formatted_text', []))):
                try:
                    full_text = batch['formatted_text'][i]
                    
                    # Find "Answer:" to create generation prompt
                    answer_splits = full_text.split("Answer:")
                    if len(answer_splits) < 2:
                        print(f"No 'Answer:' found in text {i}")
                        continue
                    
                    context_question = "Answer:".join(answer_splits[:-1]) + "Answer:"
                    
                    if i < 2:
                        print(f"Example {i} - Generating from: ...{context_question[-100:]}")
                    
                    inputs = tokenizer(
                        context_question,
                        return_tensors='pt',
                        truncation=True,
                        max_length=350,
                        padding=False
                    ).to(device)
                    
                    # FIXED: More constrained generation for classification
                    outputs = model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=max_new_tokens,  # Reduced from 30 to 5
                        min_new_tokens=1,
                        do_sample=False,  # FIXED: Use greedy decoding
                        num_beams=1,      # FIXED: No beam search
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True
                    )
                    
                    # Decode generated answer
                    full_generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
                    
                    if len(full_generated) > len(input_text):
                        generated_answer = full_generated[len(input_text):].strip()
                    else:
                        generated_answer = ""
                    
                    # FIXED: Better answer cleaning for classification
                    if generated_answer:
                        # Take only the first word (should be sentiment)
                        generated_answer = generated_answer.split()[0] if generated_answer.split() else ""
                        # Remove punctuation
                        generated_answer = generated_answer.strip('.,!?"\'').lower()
                        # Map common variations to expected labels
                        answer_mapping = {
                            'positive': 'great',
                            'negative': 'terrible', 
                            'good': 'great',
                            'bad': 'terrible',
                            'pos': 'great',
                            'neg': 'terrible'
                        }
                        generated_answer = answer_mapping.get(generated_answer, generated_answer)
                    
                    # Get ground truth
                    ground_truth_answers = get_ground_truth_answers(batch, i)
                    
                    if not ground_truth_answers:
                        print(f"No ground truth for example {i}")
                        continue
                    
                    # Calculate metrics
                    f1 = squad_f1_score(generated_answer, ground_truth_answers)
                    em = squad_exact_match(generated_answer, ground_truth_answers)
                    
                    f1_scores.append(f1)
                    em_scores.append(em)
                    
                    if i < 3:
                        print(f"Example {i}:")
                        print(f"   Generated: '{generated_answer}'")
                        print(f"   Ground truth: {ground_truth_answers}")
                        print(f"   F1: {f1:.4f}, EM: {em:.4f}")
                
                except Exception as e:
                    print(f"Example {i} failed: {e}")
                    continue
        
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
        
        print(f"Generation complete: {len(f1_scores)} valid examples")
        return avg_f1, avg_em
        
    except Exception as e:
        print(f"Generation failed: {e}")
        return 0.0, 0.0

def constrained_sentiment_generation(model, tokenizer, prompt, device):
    """
    Alternative: Constrained generation that forces valid sentiment tokens
    """
    try:
        # Define valid sentiment tokens
        sentiment_tokens = {
            'great': tokenizer.encode('great', add_special_tokens=False)[0],
            'terrible': tokenizer.encode('terrible', add_special_tokens=False)[0],
            'positive': tokenizer.encode('positive', add_special_tokens=False)[0],  
            'negative': tokenizer.encode('negative', add_special_tokens=False)[0]
        }
        
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            # Get logits for next token
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]  # Last token logits
            
            # Zero out all tokens except sentiment tokens
            constrained_logits = torch.full_like(next_token_logits, -float('inf'))
            for token_id in sentiment_tokens.values():
                constrained_logits[token_id] = next_token_logits[token_id]
            
            # Sample from constrained distribution
            probs = torch.softmax(constrained_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Decode the chosen token
            return tokenizer.decode(next_token, skip_special_tokens=True).strip()
    
    except Exception as e:
        print(f"Constrained generation failed: {e}")
        return ""


def debug_model_output(model, tokenizer, device):
    """
    Debug function to understand what the model is actually learning
    """
    model.eval()
    
    test_prompts = [
        "Context: This movie is amazing\nQuestion: What is the overall sentiment?\nAnswer:",
        "Context: This movie is awful\nQuestion: What is the overall sentiment?\nAnswer:",
        "Review: Great film\nSentiment:",
        "Sentiment of 'I love this': "
    ]
    
    print("=== MODEL DEBUG ===")
    
    for i, prompt in enumerate(test_prompts):
        print(f"\nTest {i+1}: {prompt}")
        
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        
        # Try different generation strategies
        strategies = [
            {"do_sample": False, "max_new_tokens": 3},  # Greedy
            {"do_sample": True, "temperature": 0.1, "max_new_tokens": 3},  # Low temp
            {"do_sample": True, "temperature": 1.0, "max_new_tokens": 3},  # High temp
        ]
        
        for j, params in enumerate(strategies):
            try:
                outputs = model.generate(inputs['input_ids'], **params)
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = generated[len(prompt):].strip()
                print(f"  Strategy {j+1}: '{answer}'")
            except Exception as e:
                print(f"  Strategy {j+1}: FAILED - {e}")
    
    print("\n=== TOKEN ANALYSIS ===")
    # Check if sentiment tokens are in vocabulary
    sentiment_words = ['great', 'terrible', 'positive', 'negative', 'good', 'bad']
    for word in sentiment_words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        print(f"'{word}' -> tokens: {tokens}")


def evaluate_without_generation(outputs, labels, batch, tokenizer):
    """
    Alternative evaluation that doesn't rely on generation
    Focus on next-token prediction accuracy for the answer tokens
    """
    try:
        logits = outputs.logits
        
        # Create target tokens for sentiment words
        great_tokens = tokenizer.encode(' great', add_special_tokens=False)
        terrible_tokens = tokenizer.encode(' terrible', add_special_tokens=False)
        
        prediction_accuracies = []
        
        for i in range(len(batch['formatted_text'])):
            text = batch['formatted_text'][i]
            
            # Find answer position
            if 'Answer:' not in text:
                continue
                
            # Get ground truth
            ground_truth = get_ground_truth_answers(batch, i)
            if not ground_truth:
                continue
                
            expected_answer = ground_truth[0].lower()
            
            # Get expected token sequence
            if expected_answer == 'great':
                expected_tokens = great_tokens
            elif expected_answer == 'terrible':
                expected_tokens = terrible_tokens
            else:
                continue
            
            # Find answer token positions in the sequence
            answer_start = text.find('Answer:') + len('Answer:')
            context_part = text[:answer_start]
            context_tokens = tokenizer.encode(context_part, add_special_tokens=False)
            
            # Check if model predicts correct next tokens
            start_pos = len(context_tokens) - 1  # Position before answer
            
            if start_pos < logits.shape[1] - 1:
                # Get prediction for first answer token
                predicted_token_id = torch.argmax(logits[i, start_pos, :]).item()
                expected_token_id = expected_tokens[0]
                
                accuracy = 1.0 if predicted_token_id == expected_token_id else 0.0
                prediction_accuracies.append(accuracy)
        
        return sum(prediction_accuracies) / len(prediction_accuracies) if prediction_accuracies else 0.0
        
    except Exception as e:
        print(f"Next-token evaluation failed: {e}")
        return 0.0


# Updated main metrics function
def calculate_squad_metrics(outputs, labels, batch, tokenizer, model, device):
    """
    Enhanced metrics with fallback for classification tasks
    """
    try:
        loss = outputs.loss.item() if outputs.loss is not None else 0.0
        
        # Standard token accuracy
        logits = outputs.logits
        if logits.shape[1] != labels.shape[1]:
            min_len = min(logits.shape[1], labels.shape[1])
            logits = logits[:, :min_len, :]
            labels = labels[:, :min_len]
        
        if logits.shape[1] > 1:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
        else:
            shift_logits = logits
            shift_labels = labels
        
        predictions = torch.argmax(shift_logits, dim=-1)
        answer_accuracy = calculate_answer_token_accuracy(predictions, shift_labels, batch, tokenizer)
        
        # Try generation-based evaluation
        f1_score, em_score = 0.0, 0.0
        try:
            f1_score, em_score = calculate_generation_f1_em(model, batch, tokenizer, device)
        except Exception as gen_error:
            print(f"Generation evaluation failed: {gen_error}")
            
            # Fallback: next-token prediction accuracy
            print("Using next-token prediction evaluation as fallback...")
            next_token_acc = evaluate_without_generation(outputs, labels, batch, tokenizer)
            print(f"Next-token accuracy: {next_token_acc:.4f}")
        
        return loss, answer_accuracy, f1_score, em_score
        
    except Exception as e:
        print(f"All metrics failed: {e}")
        return 0.0, 0.0, 0.0, 0.0


def test_generation_simple(model, tokenizer, device, max_new_tokens=16):
    """
    FIXED: Simple generation test with proper input handling
    """
    try:
        model.eval()
        test_input = "Question: What is the capital of France? Answer:"
        
        inputs = tokenizer(test_input, return_tensors='pt').to(device)
        
        # Handle different model wrappers
        if hasattr(model, 'generate'):
            outputs = model.generate(
                inputs['input_ids'],
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id
            )
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'generate'):
            outputs = model.base_model.generate(
                inputs['input_ids'],
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id
            )
        else:
            print("No generate method found")
            return False
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated[len(test_input):].strip()
        
        print(f"  Generation test:")
        print(f"   Input: {test_input}")
        print(f"   Generated: {answer}")
        
        return len(answer) > 0
        
    except Exception as e:
        print(f"Generation test failed: {e}")
        return False


# Legacy functions for backward compatibility
def f1_score(prediction, ground_truth):
    """Legacy function - use squad_f1_score instead"""
    return squad_f1_score(prediction, [ground_truth])


def exact_match_score(prediction, ground_truth):
    """Legacy function - use squad_exact_match instead"""
    return squad_exact_match(prediction, [ground_truth])