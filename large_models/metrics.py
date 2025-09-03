"""
Enhanced Metrics Module with F1 Score Calculation
Supports SQuAD, DROP, XSum, and SST2 tasks
"""

import torch
import numpy as np
import re
import string
from collections import Counter
import traceback

try:
    from evaluate import load as load_metric
    ROUGE_AVAILABLE = True
except ImportError:
    print("Warning: 'evaluate' library not available. ROUGE metrics will be disabled.")
    print("Install with: pip install evaluate")
    ROUGE_AVAILABLE = False


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


def calculate_token_f1(prediction, ground_truth):
    """Calculate token-level F1 score between prediction and ground truth"""
    try:
        if isinstance(ground_truth, list):
            # Handle multiple ground truth answers, take the best F1
            if not ground_truth or ground_truth[0] in ["CANNOTANSWER", "no answer"]:
                return float(normalize_answer(ground_truth[0]) == normalize_answer(prediction)) if ground_truth else 0.0
            
            all_f1s = []
            for gt in ground_truth:
                f1 = calculate_token_f1(prediction, gt)  # Recursive call for single ground truth
                all_f1s.append(f1)
            
            return max(all_f1s) if all_f1s else 0.0
        
        # Single ground truth case
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        
        if len(prediction_tokens) == 0 and len(ground_truth_tokens) == 0:
            return 1.0
        elif len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
            return 0.0
        
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        precision = num_same / len(prediction_tokens)
        recall = num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return float(f1)
        
    except Exception as e:
        print(f"Error in token F1 calculation: {e}")
        return 0.0


def calculate_exact_match(prediction, ground_truth):
    """Calculate exact match score"""
    try:
        if isinstance(ground_truth, list):
            normalized_prediction = normalize_answer(prediction)
            for gt in ground_truth:
                if normalize_answer(gt) == normalized_prediction:
                    return 1.0
            return 0.0
        else:
            return float(normalize_answer(prediction) == normalize_answer(ground_truth))
    except Exception as e:
        print(f"Error in exact match calculation: {e}")
        return 0.0


def calculate_rouge_f1(prediction, ground_truth):
    """Calculate ROUGE-L F1 score (for summarization tasks)"""
    if not ROUGE_AVAILABLE:
        print("ROUGE not available, falling back to token F1")
        return calculate_token_f1(prediction, ground_truth)
    
    try:
        rouge = load_metric('rouge')
        if isinstance(ground_truth, list):
            references = ground_truth
        else:
            references = [ground_truth]
            
        result = rouge.compute(predictions=[prediction], references=[references])
        return result['rougeL'].mid.fmeasure
        
    except Exception as e:
        print(f"ROUGE calculation failed, using token F1: {e}")
        return calculate_token_f1(prediction, ground_truth)


def calculate_classification_f1(predictions, labels, num_classes=2):
    """Calculate F1 score for classification tasks"""
    try:
        # Convert to numpy for easier manipulation
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        
        # For binary classification (SST2)
        if num_classes == 2:
            # Calculate F1 for positive class
            tp = np.sum((predictions == 1) & (labels == 1))
            fp = np.sum((predictions == 1) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))
            
            if tp + fp == 0 or tp + fn == 0:
                return 0.0
                
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            
            if precision + recall == 0:
                return 0.0
                
            f1 = 2 * (precision * recall) / (precision + recall)
            return float(f1)
        else:
            # For multi-class, calculate macro F1
            f1_scores = []
            for class_id in range(num_classes):
                tp = np.sum((predictions == class_id) & (labels == class_id))
                fp = np.sum((predictions == class_id) & (labels != class_id))
                fn = np.sum((predictions != class_id) & (labels == class_id))
                
                if tp + fp == 0 or tp + fn == 0:
                    f1_scores.append(0.0)
                    continue
                    
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                
                if precision + recall == 0:
                    f1_scores.append(0.0)
                else:
                    f1 = 2 * (precision * recall) / (precision + recall)
                    f1_scores.append(f1)
            
            return float(np.mean(f1_scores))
            
    except Exception as e:
        print(f"Classification F1 calculation failed: {e}")
        return 0.0


def extract_answer_from_generation(generated_text, input_text="", task='squad'):
    """Extract answer from generated text with improved sentiment handling"""
    try:
        # Remove input text from generation if present
        if input_text and generated_text.startswith(input_text):
            generated_answer = generated_text[len(input_text):].strip()
        else:
            generated_answer = generated_text.strip()
        
        # Task-specific answer extraction patterns
        patterns = {
            'squad': ["Answer:", "answer:"],
            'drop': ["Answer:", "answer:"], 
            'xsum': ["Summary:", "summary:"],
            'sst2': ["Sentiment:", "sentiment:"]
        }
        
        task_patterns = patterns.get(task, ["Answer:"])
        
        for pattern in task_patterns:
            if pattern in generated_answer:
                answer_part = generated_answer.split(pattern)[-1].strip()
                # Take only the first sentence/line as the answer
                answer = answer_part.split('\n')[0].split('.')[0].strip()
                # Clean up common artifacts
                answer = answer.replace('<|endoftext|>', '').replace('</s>', '').strip()
                
                # Special handling for SST2 - map various sentiment words
                if task == 'sst2':
                    answer = normalize_sentiment(answer)
                
                return answer
        
        # If no pattern found, return the first sentence
        answer = generated_answer.split('\n')[0].split('.')[0].strip()
        answer = answer.replace('<|endoftext|>', '').replace('</s>', '').strip()
        
        # Special handling for SST2 even without pattern
        if task == 'sst2':
            answer = normalize_sentiment(answer)
        
        return answer
        
    except Exception as e:
        print(f"Answer extraction failed: {e}")
        if task == 'sst2':
            return normalize_sentiment(generated_text.strip())
        return generated_text.strip()


def normalize_sentiment(text):
    """Map various sentiment words to completion words used in training (terrible/great)"""
    text = text.lower().strip()
    
    # Primary completion words from the reference template
    if 'terrible' in text:
        return 'terrible'
    elif 'great' in text:
        return 'great'
    
    # Secondary positive indicators -> map to "great"  
    positive_words = [
        'positive', 'pos', 'good', 'excellent', 'wonderful', 
        'amazing', 'fantastic', 'awesome', 'love', 'like', 'happy',
        'pleased', 'satisfied', 'joy', 'smile', 'laugh', 'funny',
        'entertaining', 'enjoyable', 'delightful', 'brilliant'
    ]
    
    # Secondary negative indicators -> map to "terrible"
    negative_words = [
        'negative', 'neg', 'bad', 'awful', 'horrible',
        'hate', 'dislike', 'sad', 'angry', 'disappointed', 'boring',
        'dull', 'stupid', 'waste', 'worst', 'sucks', 'annoying',
        'frustrating', 'depressing', 'pathetic', 'ridiculous'
    ]
    
    # Map positive words to "great"
    for word in positive_words:
        if word in text:
            return 'great'
    
    # Map negative words to "terrible"  
    for word in negative_words:
        if word in text:
            return 'terrible'
    
    # If we can't determine sentiment, return the original text
    # This will result in F1=0 but at least we can see what was generated
    return text


def calculate_generation_metrics(model, batch, tokenizer, device, task='squad', max_new_tokens=30):
    """
    Calculate generation-specific F1 and exact match scores
    Works across different tasks (SQuAD, DROP, XSum, SST2)
    """
    try:
        model.eval()
        f1_scores = []
        em_scores = []
        
        # Adjust max_new_tokens based on task
        task_tokens = {
            'squad': 20,
            'drop': 25, 
            'xsum': 50,
            'sst2': 10
        }
        max_new_tokens = task_tokens.get(task, max_new_tokens)
        
        with torch.no_grad():
            for i in range(batch['input_ids'].shape[0]):
                try:
                    # Get input text for generation with task-specific handling
                    if 'input_texts' in batch:
                        input_text = batch['input_texts'][i]
                    elif 'formatted_text' in batch:
                        full_text = batch['formatted_text'][i]
                        if task == 'squad' or task == 'drop':
                            # For QA tasks, generate from "Answer:" prompt
                            if "Answer:" in full_text:
                                input_text = full_text.split("Answer:")[0] + "Answer:"
                            else:
                                input_text = full_text
                        elif task == 'xsum':
                            if "Summary:" in full_text:
                                input_text = full_text.split("Summary:")[0] + "Summary:"
                            else:
                                input_text = full_text
                        elif task == 'sst2':
                            # SST2 uses completion format: "Sentence It was"  
                            if "It was" in full_text:
                                input_text = full_text.split("It was")[0] + "It was"
                            else:
                                # Fallback: construct from original example
                                if 'original_example' in batch and i < len(batch['original_example']):
                                    sentence = batch['original_example'][i].get('sentence', '')
                                    input_text = f"{sentence} It was"
                                else:
                                    input_text = full_text
                            if i < 3:
                                print(f"   SST2 input for generation: '{input_text}'")
                        else:
                            input_text = full_text
                    else:
                        print(f"No input text found for example {i}")
                        continue
                    
                    # Tokenize input
                    inputs = tokenizer(
                        input_text,
                        return_tensors='pt',
                        truncation=True,
                        max_length=400,  # Leave room for generation
                        padding=False
                    ).to(device)
                    
                    if inputs['input_ids'].shape[1] == 0:
                        print(f"Empty input for example {i}")
                        continue
                    
                    # Generate answer - handle wrapped models
                    if hasattr(model, 'generate'):
                        # Direct model with generate method
                        generation_model = model
                    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'generate'):
                        # Wrapped model (FullLLMModel) - use base_model
                        generation_model = model.base_model
                    else:
                        print(f"Model doesn't have generate method, skipping generation metrics")
                        continue
                    
                    # Task-specific generation parameters and constraints
                    if task == 'sst2':
                        # For sentiment classification, use more constrained generation
                        outputs = generation_model.generate(
                            inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            max_new_tokens=5,  # Very short for sentiment
                            min_new_tokens=1,
                            do_sample=False,    # Use greedy decoding for more deterministic results
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            use_cache=True
                        )
                    else:
                        # For other tasks, use the original generation parameters
                        outputs = generation_model.generate(
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
                    
                    # Decode generated text
                    full_generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Extract answer from generation
                    generated_answer = extract_answer_from_generation(full_generated, input_text, task)
                    
                    # Get ground truth with debug output
                    ground_truth = None
                    if 'original_example' in batch and i < len(batch['original_example']):
                        original_ex = batch['original_example'][i]
                        
                        # Debug output for first few examples
                        if i < 3:
                            print(f"\nDEBUG {task.upper()} Example {i}:")
                            print(f"   Original example keys: {list(original_ex.keys()) if isinstance(original_ex, dict) else 'Not a dict'}")
                            if isinstance(original_ex, dict):
                                for key in ['answers', 'answers_spans', 'answer', 'summary', 'label']:
                                    if key in original_ex:
                                        print(f"   {key}: {original_ex[key]}")
                        
                        if task == 'squad':
                            if isinstance(original_ex, dict) and 'answers' in original_ex:
                                if 'text' in original_ex['answers']:
                                    ground_truth = original_ex['answers']['text']
                                    if i < 3:
                                        print(f"   SQUAD ground truth extracted: {ground_truth}")
                        elif task == 'drop':
                            # DROP has different structure: answers_spans.spans
                            if isinstance(original_ex, dict):
                                if 'answers_spans' in original_ex:
                                    spans_data = original_ex['answers_spans']
                                    if i < 3:
                                        print(f"   DROP answers_spans: {spans_data}")
                                    if isinstance(spans_data, dict) and 'spans' in spans_data:
                                        ground_truth = spans_data['spans']
                                        if i < 3:
                                            print(f"   DROP ground truth extracted: {ground_truth}")
                                    elif isinstance(spans_data, list):
                                        ground_truth = spans_data
                                        if i < 3:
                                            print(f"   DROP ground truth (direct list): {ground_truth}")
                                elif 'answer' in original_ex:
                                    # Sometimes DROP has direct answer field
                                    ground_truth = [original_ex['answer']]
                                    if i < 3:
                                        print(f"   DROP direct answer: {ground_truth}")
                                else:
                                    # Check all keys for potential answers
                                    if i < 3:
                                        print(f"   DROP: No standard answer fields found, checking all keys:")
                                        for k, v in original_ex.items():
                                            if 'answer' in k.lower():
                                                print(f"     {k}: {v}")
                        elif task == 'xsum':
                            if isinstance(original_ex, dict) and 'summary' in original_ex:
                                ground_truth = original_ex['summary']
                                if i < 3:
                                    print(f"   XSUM ground truth extracted: {ground_truth}")
                        elif task == 'sst2':
                            if isinstance(original_ex, dict) and 'label' in original_ex:
                                # Use completion words instead of positive/negative
                                # Following the reference template mapping
                                verbalizer = {0: "terrible", 1: "great"}
                                ground_truth = verbalizer[original_ex['label']]
                                if i < 3:
                                    print(f"   SST2 ground truth extracted: {ground_truth} (from label {original_ex['label']})")
                    
                    if ground_truth is None:
                        if i < 5:  # Show more examples if no ground truth found
                            print(f"   WARNING: No ground truth found for {task} example {i}")
                        continue
                    
                    # Calculate F1 and exact match based on task
                    if task == 'xsum':
                        f1 = calculate_rouge_f1(generated_answer, ground_truth)
                        em = calculate_exact_match(generated_answer, ground_truth)
                    else:
                        f1 = calculate_token_f1(generated_answer, ground_truth)
                        em = calculate_exact_match(generated_answer, ground_truth)
                    
                    f1_scores.append(f1)
                    em_scores.append(em)
                    
                    # Debug output for first few examples
                    if i < 3:
                        print(f"{task.upper()} Generation Example {i}:")
                        print(f"   Generated: '{generated_answer}'")
                        if isinstance(ground_truth, list):
                            print(f"   Ground truth: {ground_truth[:2]}")  # Show first 2 if list
                        else:
                            print(f"   Ground truth: '{ground_truth}'")
                        print(f"   F1: {f1:.4f}, EM: {em:.4f}")
                    
                except Exception as example_error:
                    print(f"Processing failed for example {i}: {example_error}")
                    continue
        
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
        
        print(f"{task.upper()} Generation Metrics: {len(f1_scores)} valid examples")
        print(f"   Average F1: {avg_f1:.6f}, Average EM: {avg_em:.6f}")
        
        return avg_f1, avg_em
        
    except Exception as e:
        print(f"Generation metrics calculation failed: {e}")
        traceback.print_exc()
        return 0.0, 0.0


def calculate_enhanced_metrics(outputs, labels, batch, tokenizer, model, device, task='squad'):
    """
    Enhanced metrics calculation with proper F1 scoring for all tasks
    Combines forward pass metrics with generation metrics when possible
    """
    try:
        # Basic loss calculation
        loss = outputs.loss.item() if outputs.loss is not None else 0.0
        
        # Token-level accuracy calculation
        logits = outputs.logits
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
        
        predictions = torch.argmax(shift_logits, dim=-1)
        
        # Calculate token accuracy
        mask = (shift_labels != -100)
        if mask.sum() > 0:
            correct = (predictions == shift_labels) & mask
            accuracy = correct.sum().float() / mask.sum().float()
            accuracy = accuracy.item()
        else:
            accuracy = 0.0
        
        # Calculate F1 and EM using generation if we have the necessary data
        f1_score, em_score = 0.0, 0.0
        
        if 'original_example' in batch and 'formatted_text' in batch and len(batch['original_example']) > 0:
            # Use generation-based metrics for proper F1/EM calculation
            try:
                f1_score, em_score = calculate_generation_metrics(
                    model, batch, tokenizer, device, task=task
                )
            except Exception as gen_error:
                print(f"Generation metrics failed, using fallback: {gen_error}")
                # Fallback to simpler metrics if generation fails
                f1_score, em_score = 0.0, 0.0
        
        return loss, accuracy, f1_score, em_score
        
    except Exception as e:
        print(f"Enhanced metrics calculation failed: {e}")
        traceback.print_exc()
        return 0.0, 0.0, 0.0, 0.0


# Legacy functions for backward compatibility
def calculate_squad_metrics(outputs, labels, batch, tokenizer, model, device, task='squad'):
    """Legacy function name - now calls enhanced metrics"""
    return calculate_enhanced_metrics(outputs, labels, batch, tokenizer, model, device, task)


def f1_score(prediction, ground_truth):
    """Legacy function for backward compatibility"""
    return calculate_token_f1(prediction, ground_truth)


def exact_match_score(prediction, ground_truth):
    """Legacy function for backward compatibility"""
    return calculate_exact_match(prediction, ground_truth)


def calculate_client_answer_accuracy(predictions, labels, batch, tokenizer):
    """
    Client-side accuracy calculation (enhanced version)
    This version works better across different tasks while maintaining SQuAD compatibility
    """
    try:
        # If we have formatted text, try to calculate answer-specific accuracy
        if 'formatted_text' in batch and tokenizer is not None and len(batch['formatted_text']) > 0:
            accuracies = []
            
            for i in range(len(batch['formatted_text'])):
                text = batch['formatted_text'][i]
                
                # Find answer portion based on task
                patterns = ["Answer:", "Summary:", "Sentiment:"]
                answer_start = -1
                
                for pattern in patterns:
                    if pattern in text:
                        answer_start = text.find(pattern)
                        break
                
                if answer_start == -1:
                    continue
                    
                context_part = text[:answer_start + len(pattern.split(':')[0]) + 1]
                answer_text = text[answer_start + len(pattern.split(':')[0]) + 1:].strip()
                
                if not answer_text:
                    continue
                
                # Tokenize parts
                context_tokens = tokenizer.encode(context_part, add_special_tokens=False)
                answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
                
                if len(answer_tokens) == 0:
                    continue
                
                start_idx = len(context_tokens)
                end_idx = min(start_idx + len(answer_tokens), predictions.shape[1])
                
                if start_idx < end_idx and start_idx < predictions.shape[1]:
                    answer_preds = predictions[i, start_idx:end_idx]
                    answer_labels = labels[i, start_idx:end_idx]
                    
                    if len(answer_preds) > 0:
                        correct = (answer_preds == answer_labels).sum().item()
                        total = len(answer_preds)
                        accuracies.append(correct / total)
            
            if accuracies:
                return sum(accuracies) / len(accuracies)
        
        # Fallback to overall accuracy
        mask = (labels != -100) & (labels != 0)
        
        if mask.sum() == 0:
            # Handle prefix tuning case
            prefix_len = 5  # Assume 5 prefix tokens
            if labels.shape[1] > prefix_len:
                start_idx = prefix_len
                end_idx = min(labels.shape[1] - 1, start_idx + 100)  # Look at next 100 tokens
                relevant_labels = labels[:, start_idx:end_idx]
                relevant_preds = predictions[:, start_idx:end_idx]
                mask = (relevant_labels != -100) & (relevant_labels != 0)
                
                if mask.sum() > 0:
                    correct = (relevant_preds == relevant_labels) & mask
                    return correct.sum().float() / mask.sum().float()
            
            return 0.0
        else:
            correct = (predictions == labels) & mask
            return correct.sum().float() / mask.sum().float()
        
    except Exception as e:
        print(f"Client accuracy calculation failed: {e}")
        return 0.0


if __name__ == "__main__":
    # Test the enhanced metrics
    print("Testing enhanced metrics...")
    
    # Test token F1
    pred = "The quick brown fox"
    gt = "quick brown fox jumps"
    f1 = calculate_token_f1(pred, gt)
    print(f"Token F1 test: {f1:.4f}")
    
    # Test classification F1 
    preds = np.array([1, 0, 1, 1, 0])
    labels = np.array([1, 0, 0, 1, 0])
    clf_f1 = calculate_classification_f1(preds, labels)
    print(f"Classification F1 test: {clf_f1:.4f}")
    
    print("Enhanced metrics test completed!")