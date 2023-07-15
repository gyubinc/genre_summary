def summary(tokenizer, model, passage):
    input_ids = tokenizer.encode(passage, return_tensors='pt', add_special_tokens=True)
    outputs = model.generate(input_ids=input_ids, num_beams=8)
    decoded_summary = tokenizer.decode(outputs.squeeze(), skip_special_tokens=True)
    return decoded_summary

