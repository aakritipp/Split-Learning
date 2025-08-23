"""
SQUAD Metrics Calculation Module
Handles all metric calculations for SQUAD question answering task
Based on official SQUAD evaluation script and MeZO implementation
"""

import torch
import numpy as np
import re
import string
from collections import Counter
from tqdm import tqdm


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


def squad_f1_score(prediction, ground_truth_list):
    """
    Calculate F1 score for SQUAD - handles multiple ground truth answers
    Args:
        prediction: str - the predicted answer
        ground_truth_list: list of str - list of possible correct answers
    """
    try:
        # Handle the case where ground_truth_list is a single string
        if isinstance(ground_truth_list, str):
            ground_truth_list = [ground_truth_list]
        
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
        print(f"❌ F1 calculation error: {e}")
        return 0.0


def squad_exact_match(prediction, ground_truth_list):
    """
    Calculate exact match for SQUAD - handles multiple ground truth answers
    Args:
        prediction: str - the predicted answer  
        ground_truth_list: list of str - list of possible correct answers
    """
    try:
        # Handle the case where ground_truth_list is a single string
        if isinstance(ground_truth_list, str):
            ground_truth_list = [ground_truth_list]
        
        # Check exact match against any of the ground truth answers
        normalized_prediction = normalize_answer(prediction)
        
        for ground_truth in ground_truth_list:
            normalized_ground_truth = normalize_answer(ground_truth)
            if normalized_prediction == normalized_ground_truth:
                return 1.0
        
        return 0.0
        
    except Exception as e:
        print(f"❌ EM calculation error: {e}")
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
        print(f"❌ Answer token accuracy failed: {e}")
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
        print(f"❌ Client answer accuracy calculation failed: {e}")
        return 0.0


def calculate_generation_f1_em(model, batch, tokenizer, device, max_new_tokens=30):
    """Generate answers and calculate F1/EM - UPDATED TO MATCH REFERENCE"""
    try:
        model.eval()
        f1_scores = []
        em_scores = []
        
        print(f"\nGeneration Debug: Processing {len(batch.get('formatted_text', []))} examples")
        
        with torch.no_grad():
            for i in range(len(batch.get('formatted_text', []))):
                try:
                    # Get the full formatted text
                    full_text = batch['formatted_text'][i]
                    
                    # Find the LAST occurrence of "Answer:"
                    answer_splits = full_text.split("Answer:")
                    if len(answer_splits) < 2:
                        print(f"⚠️ No 'Answer:' found in text {i}")
                        continue
                    
                    # Reconstruct context + question + "Answer:"
                    context_question = "Answer:".join(answer_splits[:-1]) + "Answer:"
                    
                    # Debug output
                    if i < 2:
                        print(f"🔍 Example {i} - Generating from (last 100 chars): ...{context_question[-100:]}")
                    
                    # Tokenize and generate
                    inputs = tokenizer(
                        context_question,
                        return_tensors='pt',
                        truncation=True,
                        max_length=350,
                        padding=False
                    ).to(device)
                    
                    if inputs['input_ids'].shape[1] == 0:
                        print(f"⚠️ Empty input for example {i}")
                        continue
                    
                    # Generate answer
                    outputs = model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=1,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
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
                        generated_answer = full_generated.strip()
                    
                    # Clean up the generated answer
                    if generated_answer:
                        generated_answer = generated_answer.split('\n')[0].split('.')[0]
                        # Remove common artifacts
                        generated_answer = generated_answer.replace('<|endoftext|>', '').strip()
                    
                    # Get ground truth answers as list (SQUAD can have multiple correct answers)
                    ground_truth_answers = []
                    if 'original_example' in batch and i < len(batch['original_example']):
                        original_ex = batch['original_example'][i]
                        if (isinstance(original_ex, dict) and 
                            'answers' in original_ex and 
                            isinstance(original_ex['answers'], dict) and
                            'text' in original_ex['answers']):
                            # SQUAD answers is a list - use all of them
                            ground_truth_answers = original_ex['answers']['text']
                    
                    if not ground_truth_answers:
                        print(f"⚠️ No ground truth answers for example {i}")
                        continue
                    
                    # Calculate F1 and EM with multiple ground truth answers
                    f1 = squad_f1_score(generated_answer, ground_truth_answers)
                    em = squad_exact_match(generated_answer, ground_truth_answers)
                    
                    # Validate scores
                    if not (0 <= f1 <= 1) or not (0 <= em <= 1):
                        print(f"⚠️ Invalid scores for example {i}: F1={f1}, EM={em}")
                        continue
                    
                    f1_scores.append(f1)
                    em_scores.append(em)
                    
                    # Debug output for first few examples
                    if i < 3:
                        print(f"🔍 Example {i}:")
                        print(f"   Generated: '{generated_answer}'")
                        print(f"   Ground truth: {ground_truth_answers}")
                        print(f"   F1: {f1:.4f}, EM: {em:.4f}")
                    
                except Exception as example_error:
                    print(f"⚠️ Processing failed for example {i}: {example_error}")
                    continue
        
        # Calculate averages
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
        
        print(f"Generation complete: {len(f1_scores)} valid examples out of {len(batch.get('formatted_text', []))}")
        print(f"Average F1: {avg_f1:.6f}, Average EM: {avg_em:.6f}")
        
        return avg_f1, avg_em
        
    except Exception as e:
        print(f"❌ Generation F1/EM calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0


def calculate_squad_metrics(outputs, labels, batch, tokenizer, model, device):
    """Calculate SQUAD-specific metrics - MAIN FUNCTION"""
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


def test_generation_simple(model, tokenizer, device):
    """Simple generation test to verify model works"""
    try:
        test_input = "Question: What is the capital of France? Answer:"
        
        inputs = tokenizer(test_input, return_tensors='pt').to(device)
        
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated[len(test_input):].strip()
        
        print(f"  Generation test:")
        print(f"   Input: {test_input}")
        print(f"   Generated: {answer}")
        
        return len(answer) > 0
        
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        return False


# Legacy functions for backward compatibility
def f1_score(prediction, ground_truth):
    """Legacy function - use squad_f1_score instead"""
    return squad_f1_score(prediction, [ground_truth])


def exact_match_score(prediction, ground_truth):
    """Legacy function - use squad_exact_match instead"""
    return squad_exact_match(prediction, [ground_truth])