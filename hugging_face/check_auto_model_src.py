import inspect
from transformers import AutoModelForQuestionAnswering

# 加载预训练模型
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

# 获取并打印 __init__ 方法的源码
init_code = inspect.getsource(model.__init__)
print(init_code)

forward_code = inspect.getsource(model.forward)
print(forward_code)