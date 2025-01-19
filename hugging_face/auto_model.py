from transformers import AutoTokenizer, AutoModel
import torch

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

input = "The capital of [MASK] is Pairs."

tokenizer_input = tokenizer(input, return_tensors="pt")

# 获取预测
with torch.no_grad():
    outputs = model(**tokenizer_input)
    
last_hidden_states = outputs.last_hidden_state

print(f"Last hidden state shape: {last_hidden_states.shape}")