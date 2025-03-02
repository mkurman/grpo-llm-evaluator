import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
from transformers import get_cosine_schedule_with_warmup
import bitsandbytes as bnb
from datetime import datetime
from tqdm import tqdm
from accelerate import Accelerator
import asyncio

from grpo_llm_eval.func.model_utils import load_student_model, load_teacher_model
from grpo_llm_eval.func.data_utils import load_dataset_function
from grpo_llm_eval.func.generation_utils import generate_response
from grpo_llm_eval.func.evaluation_utils import evaluate_response
from grpo_llm_eval.func.trl_utils import combine_and_train
from grpo_llm_eval.func.config_utils import load_config
from grpo_llm_eval.args import parse_args
from logging import getLogger
import warnings

warnings.filterwarnings("ignore")

logger = getLogger(__name__)


async def main():
    args = parse_args()
    config = load_config(args.config)
    current_date = datetime.now().strftime("%Y-%m-%d %H")

    if current_date not in config.output_dir:
        config.output_dir = (
            f"{config.output_dir}/{datetime.now().strftime('%Y-%m-%d %H_%M_%S')}"
        )
    else:
        config.output_dir = config.output_dir

    tokenizer, student_model = load_student_model(
        config, load_in_4bit=config.load_in_4bit
    )
    _, ref_model = load_student_model(config, load_in_4bit=True)

    dataset = load_dataset_function(config).shuffle(seed=config.seed)

    dataset = dataset.rename_columns(
        {
            config.dataset_question_column: "question",
            config.dataset_answer_column: "answer",
        }
    )

    print(dataset)

    teacher_model = load_teacher_model(config)  # Now this is just the model name

    optimizer = bnb.optim.AdamW8bit(student_model.parameters(), lr=config.learning_rate)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.total_steps,
    )

    accelerator = Accelerator()

    student_model, ref_model, optimizer, dataset, scheduler = accelerator.prepare(
        student_model, ref_model, optimizer, dataset, scheduler
    )

    logger.debug("Starting training loop...")

    total_steps = config.total_steps * config.accumulation_steps

    step_bar = tqdm(range(total_steps), total=total_steps, desc="Training", position=0)

    step_bar.update(1)

    student_model.train()

    for i in range(total_steps):
        # Get data from dataset
        index = i % len(dataset)
        student_responses = generate_response(
            student_model, tokenizer, dataset["question"][index], config
        )

        teacher_feedback = await evaluate_response(
            teacher_model, student_responses, dataset["answer"][index], config
        )

        reward, advantage, policy_loss, sft_loss = combine_and_train(
            student_model,
            ref_model,
            tokenizer,
            teacher_feedback,
            dataset["question"][index],
            student_responses,
            config.output_dir,
            config,
        )

        loss = policy_loss + sft_loss * config.sft_beta

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), config.max_grad_norm)

        if config.accumulation_steps == 1 or (
            i % config.accumulation_steps == 0 and i > 0
        ):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            step_bar.update(int(i / config.accumulation_steps) + 1)

        step_bar.set_postfix(
            loss=loss.item(),
            reward=reward,
            advantage=advantage,
            policy_loss=policy_loss.item(),
            sft_loss=sft_loss.item(),
            lr=scheduler.get_last_lr()[0],
        )

        if i / config.accumulation_steps % config.save_steps == 0 and i > 0:
            student_dtype = student_model.config.torch_dtype

            path = os.path.join(output_dir, f"checkpoint-{i}")
            accelerator.unwrap_model(student_model).save_pretrained(
                path, state_dict=student_model.state_dict(), safe_serialization=True
            )

            student_model = accelerator.prepare(student_model)

            # Workaround to fix the dtype of the model (unsloth issue)
            student_model.config.torch_dtype = student_dtype

            logger.debug(f"Model saved to {path}")

            # Delete previous checkpoints
            if i / config.accumulation_steps > config.save_steps:
                prev_path = os.path.join(
                    output_dir, f"checkpoint-{i-config.save_steps}"
                )
                os.system(f"rm -rf {prev_path}")
                logger.debug(f"Deleted checkpoint {prev_path}")

        if i / config.accumulation_steps % config.total_steps == 0 and i > 0:
            student_model.save_pretrained(os.path.join(output_dir, "final"))
            logger.debug(f"Final model saved to {os.path.join(output_dir, 'final')}")
            step_bar.update()
            break


if __name__ == "__main__":
    pass


def run_main():
    try:
        asyncio.run(main(), debug=True)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
