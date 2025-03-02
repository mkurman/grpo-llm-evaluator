from unsloth import FastLanguageModel


def generate_response(student_model, tokenizer, input_text, config):
    """
    Generates a response from the student model given an input text.

    Args:
        student_model: The student model to use for generating the response.
        tokenizer: The tokenizer to use for encoding and decoding text.
        input_text (str): The input text to generate a response for.
        config: The configuration object containing hyperparameters.

    Returns:
        list: A list of generated responses.
    """
    prompt = [{"role": "user", "content": input_text}]

    if config.system_prompt is not None:
        system_prompt = [{"role": "system", "content": config.system_prompt}]
        prompt = system_prompt + prompt

    prompt = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
        padding=True,
        truncation=True,
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    if config.use_unsloth:
        student_model = FastLanguageModel.for_inference(student_model)
    else:
        student_model = student_model.to("cuda")
        student_model.eval()
    # streamer = TextStreamer(tokenizer=tokenizer)
    outputs = student_model.generate(
        **inputs,
        max_new_tokens=config.max_new_tokens,
        num_return_sequences=config.num_return_sequences,
        do_sample=True,
        use_cache=True,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        # streamer=streamer,
    )
    outputs = outputs.to("cpu")
    inputs = inputs.to("cpu")

    outputs = outputs[:, inputs.input_ids.shape[1] :]

    responses = []
    for output in outputs:
        response = tokenizer.decode(output, skip_special_tokens=True)
        responses.append(response.replace(prompt, "").strip())

    return responses
