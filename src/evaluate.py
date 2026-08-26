"""Generation helpers shared by the qualitative checks and the RR/Over-RR scoring pipeline."""


def generate_samples(model, tokenizer, prompts: list[str], **generation_kwargs) -> list[str]:
    """Generate one completion per prompt in `prompts`, using an already-loaded `model`/`tokenizer`."""
    completions = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        output = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            **generation_kwargs,
        )
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        completions.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return completions


def generate_batch(model, tokenizer, prompts: list[str], **generation_kwargs) -> list[str]:
    """Generate completions for all `prompts` at once.

    Decoder-only models must be left-padded for batched generation: the next
    token is always predicted from the last position, so padding on the right
    would push that position past the real content.
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    output = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        **generation_kwargs,
    )
    prompt_length = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(output[i][prompt_length:], skip_special_tokens=True)
        for i in range(len(prompts))
    ]
