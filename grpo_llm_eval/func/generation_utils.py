from unsloth import FastLanguageModel

def generate_response(student_model, tokenizer, input_text, config):
    prompt = [{"role": "user", "content": input_text}]
    prompt = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True, padding=True, truncation=True
        )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    student_model = FastLanguageModel.for_inference(student_model)
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
    
    outputs = outputs[:, inputs.input_ids.shape[1]:]

    responses = []
    for output in outputs:
        response = tokenizer.decode(output, skip_special_tokens=True)
        responses.append(response.replace(prompt, "").strip())

    return responses
