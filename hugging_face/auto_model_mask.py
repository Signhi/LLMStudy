from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

input = "The capital of [MASK] is Pairs."

tokenizer_input = tokenizer(input, return_tensors="pt")

# 获取预测
with torch.no_grad():
    outputs = model(**tokenizer_input)
    print(f"Last hidden state shape: {outputs.last_hidden_state.shape}")
    predictions = outputs.logits

# 获取最高得分的预测词
masked_index = (tokenizer_input.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
predicted_token_id = predictions[0, masked_index].argmax(dim=-1).item()
predicted_token = tokenizer.decode([predicted_token_id])

print(f"预测结果: {predicted_token}")