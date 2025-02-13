import torch
import numpy as np
import os
import json
from unsloth import FastLanguageModel

# TRL HuggingFace implementation
# Source: https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md
def get_per_token_logps(model, input_ids, attention_mask, logits_to_keep):
    # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
    logits = model(
        input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1
    ).logits  # (B, L, V)
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

    input_ids = input_ids[:, -logits_to_keep:]
    # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
    # See https://github.com/huggingface/trl/issues/2770
    logits = logits[:, -logits_to_keep:]

    # Compute the log probabilities for the input tokens.
    token_logits = logits.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    # use a loop to reduce memory peak
    logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
    token_log_probs = token_logits - logsumexp_values  # log_softmax = logits - log(sum(exp(logits)))
    return token_log_probs

# TRL HuggingFace implementation
# Source: https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md
def compute_loss(model, ref_per_token_logps, inputs, advantages, logits_to_keep, beta: float = 0.05):

    completion_mask = inputs['labels'].ne(-100).float()[:, -logits_to_keep:]

    per_token_logps = get_per_token_logps(model, inputs['input_ids'], inputs['attention_mask'], logits_to_keep)

    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    return loss

def grpo(
    student_model,
    ref_model,
    tokenizer,
    teacher_feedbacks,
    original_input,
    original_responses,
    config,
):
    combined_outputs = [[
        {"role": "user", "content": original_input},
        {"role": "assistant", "content": original_response},
    ] for original_response in original_responses]

    tokenized_inputs = tokenizer(
        tokenizer.apply_chat_template(
            combined_outputs, tokenize=False, add_generation_prompt=True
        ),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    inputs = tokenized_inputs["input_ids"]
    labels = inputs.clone()

    logits_to_keep = []

    # Mask every tokens except the last assistant answer
    for i in range(len(combined_outputs)):
        combined_output_to_mask = [
            {"role": "user", "content": original_input},
        ]

        tokenized_inputs_to_mask = tokenizer(
            tokenizer.apply_chat_template(
                combined_output_to_mask, tokenize=False, add_generation_prompt=False
            ),
            return_tensors="pt",
            padding=False,
            truncation=False,
        )

        logits_to_keep.append(len(tokenized_inputs_to_mask["input_ids"][0]))

        labels[i, :len(tokenized_inputs_to_mask["input_ids"][0])] = -100

    tokenized_inputs['labels'] = labels

    tokenized_inputs = tokenized_inputs.to(student_model.device)

    student_model = FastLanguageModel.for_training(student_model)

    ref_model = FastLanguageModel.for_inference(ref_model)
    logits_to_keep = inputs.size(1) - np.max(logits_to_keep)
    
    with torch.inference_mode():
        ref_per_token_logps = get_per_token_logps(
            ref_model, tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], logits_to_keep
        )

    from func.evaluation_utils import extract_scores
    rewards = []

    for teacher_feedback in teacher_feedbacks:
        thought_score, answer_score, style_score = extract_scores(teacher_feedback)
        reward = (thought_score * 0.07 + answer_score * 0.1 + style_score * 0.03) / 2.0
        rewards.append(torch.tensor(reward))

    rewards = torch.stack(rewards).to(student_model.device).float()

    # Compute grouped-wise rewards
    mean_grouped_rewards = rewards.view(-1, config.num_return_sequences).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, config.num_return_sequences).std(dim=1)

    # Normalize the rewards to compute the advantages
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(config.num_return_sequences, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(config.num_return_sequences, dim=0)
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
    
    # Policy gradient loss with advantage
    policy_loss = compute_loss(student_model, ref_per_token_logps, tokenized_inputs, advantages, logits_to_keep=logits_to_keep, beta=config.grpo_beta)

    policy_loss = policy_loss / logits_to_keep / config.num_return_sequences

    return rewards, advantages, policy_loss

def sft_on_eval(
    student_model,
    tokenizer,
    teacher_feedbacks,
    original_input,
    original_responses,
):
    combined_outputs = [[
        {"role": "user", "content": original_input},
        {"role": "assistant", "content": original_response},
        {"role": "user", "content": "Let's recheck the previous answer"},
        {"role": "assistant", "content": teacher_feedback},
    ] for original_response, teacher_feedback in zip(original_responses, teacher_feedbacks)]

    tokenized_inputs = tokenizer(
        tokenizer.apply_chat_template(
            combined_outputs, tokenize=False, add_generation_prompt=True
        ),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    inputs = tokenized_inputs["input_ids"]
    labels = inputs.clone()

    logits_to_keep = []

    # Mask every tokens except the last assistant answer
    for i in range(len(combined_outputs)):
        combined_output_to_mask = [
            {"role": "user", "content": original_input},
            {"role": "assistant", "content": original_responses[i]},
            {"role": "user", "content": "Let's recheck the previous answer"},
        ]

        tokenized_inputs_to_mask = tokenizer(
            tokenizer.apply_chat_template(
                combined_output_to_mask, tokenize=False, add_generation_prompt=False
            ),
            return_tensors="pt",
            padding=False,
            truncation=False,
        )

        logits_to_keep.append(len(tokenized_inputs_to_mask["input_ids"][0]))

        labels[i, :len(tokenized_inputs_to_mask["input_ids"][0])] = -100

    tokenized_inputs['labels'] = labels

    tokenized_inputs = tokenized_inputs.to(student_model.device)

    student_model = FastLanguageModel.for_training(student_model)
    loss = student_model(**tokenized_inputs).loss

    return loss
    

def combine_and_train(
    student_model,
    ref_model,
    tokenizer,
    teacher_feedbacks,
    original_input,
    original_responses,
    output_dir,
    config,
):
    
    rewards, advantages, policy_loss = grpo(
        student_model,
        ref_model,
        tokenizer,
        teacher_feedbacks,
        original_input,
        original_responses,
        config,
    )

    if config.sft_beta > 0:
        sft_loss = sft_on_eval(
            student_model,
            tokenizer,
            teacher_feedbacks,
            original_input,
            original_responses,
        )
    else:
        sft_loss = torch.tensor(0.0).to(student_model.device)

    # Prepare data for saving
    for i, (original_response, teacher_feedback) in enumerate(zip(original_responses, teacher_feedbacks)):
        combined_output = {
            "input": original_input,
            "original_response": original_response,
            "teacher_feedback": teacher_feedback,
            "reward": rewards[i].item(),
            "advantage": advantages[i].item(),
            "policy_loss": policy_loss.item(),
            "sft_loss": sft_loss.item(),
        }

        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Define the file path
        file_path = os.path.join(output_dir, "responses.jsonl")

        # Append the data to the JSONL file
        with open(file_path, "a") as f:
            json.dump(combined_output, f)
            f.write("\n")  # Add newline to separate JSON objects
    
    reward = torch.sum(rewards).item()
    advantage = advantages.mean().item()

    return reward, advantage, policy_loss, sft_loss
