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
            # NOTE: predictions/labels are computed from shifted logits/labels.
            # The first answer token at position `start_idx` in the unshifted sequence
            # corresponds to index `start_idx-1` in the shifted tensors.
            start_idx = len(context_tokens)
            end_idx = start_idx + len(answer_tokens)

            # Align indices for shifted tensors
            start_shifted = max(start_idx - 1, 0)
            end_shifted = max(end_idx - 1, 0)

            if end_shifted <= predictions.shape[1] and end_shifted <= labels.shape[1] and end_shifted > start_shifted:
                answer_preds = predictions[i, start_shifted:end_shifted]
                answer_labels = labels[i, start_shifted:end_shifted]

                if answer_preds.numel() > 0:
                    correct = (answer_preds == answer_labels).sum().item()
                    total = int(answer_preds.numel())
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
    Task-aware evaluation helper.
    - For classification tasks (sst2): single-step constrained classification at first answer token.
    - For generation tasks: short greedy generation and EM/F1.
    """
    model.eval()
    is_classification = 'class_label' in batch
    f1_scores = []
    em_scores = []

    try:
        if is_classification:
            print(f"\nClassification Debug: Processing {len(batch.get('formatted_text', []))} examples")
        else:
            print(f"\nGeneration Debug: Processing {len(batch.get('formatted_text', []))} examples")

        with torch.no_grad():
            for i in range(len(batch.get('formatted_text', []))):
                full_text = batch['formatted_text'][i]
                if 'Answer:' not in full_text:
                    continue

                # Build prompt up to 'Answer:'
                parts = full_text.split('Answer:')
                context_question = 'Answer:'.join(parts[:-1]) + 'Answer:'

                if is_classification:
                    # next-token classification between ' great' vs ' terrible'
                    prompt = context_question + ' '
                    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=350, padding=False).to(device)
                    out = model(**inputs)
                    next_logits = out.logits[0, -1, :]

                    tok_great = tokenizer.encode(' great', add_special_tokens=False) or tokenizer.encode('great', add_special_tokens=False)
                    tok_terr = tokenizer.encode(' terrible', add_special_tokens=False) or tokenizer.encode('terrible', add_special_tokens=False)
                    if not tok_great or not tok_terr:
                        continue
                    id_great = tok_great[0]
                    id_terr = tok_terr[0]
                    logit_great = float(next_logits[id_great].item())
                    logit_terr = float(next_logits[id_terr].item())
                    predicted = 'great' if logit_great >= logit_terr else 'terrible'

                    gold = get_ground_truth_answers(batch, i)
                    if not gold:
                        continue
                    f1 = squad_f1_score(predicted, gold)
                    em = squad_exact_match(predicted, gold)
                    f1_scores.append(f1)
                    em_scores.append(em)
                    if i < 3:
                        print(f"Example {i}:")
                        print(f"   Predicted: '{predicted}' (logits: great={logit_great:.2f}, terrible={logit_terr:.2f})")
                        print(f"   Ground truth: {gold}")
                        print(f"   F1: {f1:.4f}, EM: {em:.4f}")
                else:
                    # generation path
                    # Ensure suffix with 'Answer:' is preserved like training (left truncation)
                    _old_side = getattr(tokenizer, 'truncation_side', 'right')
                    try:
                        tokenizer.truncation_side = 'left'
                    except Exception:
                        pass
                    inputs = tokenizer(context_question, return_tensors='pt', truncation=True, max_length=350, padding=False).to(device)
                    # restore truncation side
                    try:
                        tokenizer.truncation_side = _old_side
                    except Exception:
                        pass
                    gen = model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs.get('attention_mask', None),
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=1,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
                    full_generated = tokenizer.decode(gen[0], skip_special_tokens=True)
                    input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
                    generated_answer = full_generated[len(input_text):].strip() if len(full_generated) > len(input_text) else ''
                    # Keep multi-token span; normalize lightly by trimming punctuation and stopping at newline
                    if generated_answer:
                        generated_answer = generated_answer.split('\n')[0].strip('.,!?!"\'').lower()
                    gold = get_ground_truth_answers(batch, i)
                    if not gold:
                        continue
                    f1 = squad_f1_score(generated_answer, gold)
                    em = squad_exact_match(generated_answer, gold)
                    f1_scores.append(f1)
                    em_scores.append(em)
                    if i < 3:
                        print(f"Example {i}:")
                        print(f"   Generated: '{generated_answer}'")
                        print(f"   Ground truth: {gold}")
                        print(f"   F1: {f1:.4f}, EM: {em:.4f}")

        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
        if is_classification:
            print(f"Classification complete: {len(f1_scores)} valid examples")
        else:
            print(f"Generation complete: {len(f1_scores)} valid examples")
        return avg_f1, avg_em

    except Exception as e:
        print(f"Evaluation failed: {e}")
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
                # Preserve suffix during debug tokenization as well
                _old_side = getattr(tokenizer, 'truncation_side', 'right')
                try:
                    tokenizer.truncation_side = 'left'
                except Exception:
                    pass
                outputs = model.generate(inputs['input_ids'], **params)
                try:
                    tokenizer.truncation_side = _old_side
                except Exception:
                    pass
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
def calculate_squad_metrics(outputs, labels, batch, tokenizer, model, device, generation_max_new_tokens: int = 5):
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
            f1_score, em_score = calculate_generation_f1_em(
                model, batch, tokenizer, device, max_new_tokens=generation_max_new_tokens
            )
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