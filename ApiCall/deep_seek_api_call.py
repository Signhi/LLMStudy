# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
import os 


client = OpenAI(api_key="sk-6869a1bf0b2f449f95b23627db6e965a", base_url="https://api.deepseek.com")
messages=[
            {"role": "system", "content": "You are a helpful assistant"}
        ]

chatNum = 3

# 单轮对话
def GetResponse(content):
    messages.append(content)
    response = client.chat.completions.create(
        messages=messages,
        model="deepseek-chat",
        stream=False,
        # stream_options={"include_usage": True}
    )
    print(response.choices[0].message.content)

# 多轮对话
def GetMulResponse(chatNum):
    for index in range(chatNum):
        content = input("请输入：")
        messages.append({'role': 'user', 'content': content})
        assistantOutput = client.chat.completions.create(
            messages=messages,
            model="deepseek-chat",
            stream=False
        )
        messages.append({'role': 'assistant', 'content': assistantOutput.choices[0].message.content})
        print(f'模型输出：{assistantOutput.choices[0].message.content}')
            
if __name__ == "__main__":
    GetMulResponse(chatNum)

