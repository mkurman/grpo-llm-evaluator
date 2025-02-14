import re
import asyncio
import os
import json
import aiohttp
from logging import getLogger

logger = getLogger(__name__)


async def evaluate_response(
    teacher_model_name, student_responses, ground_truth, config
):
    """
    Evaluates the student responses using the teacher model.

    Args:
        teacher_model_name (str): The name of the teacher model to use for evaluation.
        student_responses (list): A list of student responses to evaluate.
        ground_truth (str): The ground truth response to compare the student responses to.
        config: The configuration object containing hyperparameters.
    
    Returns:
        list: A list of teacher evaluations for the student responses.
    """

    prompts = []

    openai_key = os.environ.get("OPENAI_API_KEY")

    for student_response in student_responses:
        evaluation_prompt = (
            "You are a teacher evaluating a student's answer. "
            "Evaluate the following student's response in two parts:\n"
            "1) Correctness of the thought process (rated from 1 to 10), and\n"
            "2) Correctness of the final answer (rated from 1 to 10).\n"
            "3) Format and clarity of the response (rated from 1 to 10). Student must have <think> and </think> tags included! If not, give 0 points for this assesment.\n\n"
            "Think about the student's thought process and the final answer using <think> ... </think> tags.\n"
            "Provide detailed feedback after your thinking process using the following XML format:\n"
            "<evaluation>\n"
            "  <thought_process><score>{score}</score><explanation>{explanation}</thought_process>\n"
            "  <answer><score>{score}</score><explanation>{explanation}</answer>\n"
            "  <format><score>{score}</score><explanation>{explanation}</format>\n"
            "</evaluation>\n\n"
            f"Student Response:\n{student_response}\n"
            f"Ground Truth:\n{ground_truth}\n\n"
        )

        prompts.append([{"role": "user", "content": evaluation_prompt}])

    teacher_outputs = []

    async def call_openai(prompt):
        url = os.path.join(config.openai_base_url, 'chat', 'completions')
        headers = {
            "Authorization": f"Bearer {openai_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": teacher_model_name,
            "messages": prompt,
            "max_tokens": config.max_feedback_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "include_reasoning": True,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, headers=headers, data=json.dumps(payload)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = ''

                        if 'reasoning' in data['choices'][0]['message']:
                            content = f"<think>\n{data['choices'][0]['message']['reasoning'].strip()}\n</think>\n"

                        return f"{content}{data["choices"][0]["message"]["content"]}".strip()
                    else:
                        logger.error(
                            f"OpenAI API request failed with status: {response.status}"
                        )
                        return None
        except Exception as e:
            logger.error(f"Error during OpenAI API request: {e}")
            return None

    tasks = [call_openai(prompt) for prompt in prompts]
    responses = await asyncio.gather(*tasks)

    for response in responses:
        if response:
            teacher_outputs.append(response)
        else:
            teacher_outputs.append(None)

    for i, teacher_output in enumerate(teacher_outputs):
        if teacher_output is None:
            teacher_outputs[i] = (
                "<evaluation><thought_process><score>5</score><explanation>API Error</explanation></thought_process><answer><score>5</score><explanation>API Error</explanation></answer><style><score>5</score><explanation>API Error</explanation></style></evaluation>"
            )
        else:
            if "<think>" not in teacher_output:
                teacher_outputs[i] = f"<think>\n{teacher_output.strip()}".strip()

            teacher_outputs[i] = teacher_outputs[i].strip()

    return teacher_outputs


def extract_scores(teacher_feedback):
    """
    Extracts the thought and answer scores from the teacher's feedback by using regex.
    """
    if not teacher_feedback:
        logger.debug("Teacher feedback is empty.")
        return 5, 5

    # Use a regex to capture everything between <evaluation> and </evaluation>
    eval_pattern = re.compile(r"<evaluation>(.*?)</evaluation>", flags=re.DOTALL)
    match = eval_pattern.search(teacher_feedback)
    if not match:
        logger.debug(
            "Could not find complete <evaluation>...</evaluation> in feedback."
        )
        return 5, 0.5, 5

    # Extract just the content of the <evaluation> tag
    evaluation_content = match.group(1)

    # Regex for <thought_process><score>...</score>
    thought_score_pattern = re.compile(
        r"<thought_process>.*?<score>\s*(\d+)\s*</score>.*?<explanation>", re.DOTALL
    )
    # Regex for <answer><score>...</score>
    answer_score_pattern = re.compile(
        r"<answer>.*?<score>\s*(\d+\.?\d*)\s*</score>.*?<explanation>", re.DOTALL
    )

    style_score_pattern = re.compile(
        r"<format>.*?<score>\s*(\d+)\s*</score>.*?<explanation>", re.DOTALL
    )

    # Default scores
    thought_score = 5
    answer_score = 5
    style_score = 5

    # Search for thought_process score
    t_match = thought_score_pattern.search(evaluation_content)
    if t_match:
        # Convert the captured group to int
        thought_score = int(t_match.group(1).strip())
    else:
        logger.debug("Could not find thought_process score.")

    # Search for answer score
    a_match = answer_score_pattern.search(evaluation_content)
    if a_match:
        answer_score = int(a_match.group(1).strip())
    else:
        logger.debug("Could not find answer score.")

    s_match = style_score_pattern.search(evaluation_content)
    if s_match:
        style_score = int(s_match.group(1).strip())
    else:
        logger.debug("Could not find format score.")

    return thought_score, answer_score, style_score
