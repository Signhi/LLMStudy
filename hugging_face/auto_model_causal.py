from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

input = "Hello, how old are you?"

tokenizer_input = tokenizer(input, return_tensors="pt")

outputs = model.generate(**tokenizer_input, max_length=50, do_sample=True, top_p=0.95, temperature=0.7)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))


