import openai
import gradio as gr
import os
import json
from typing import List, Dict, Tuple

OPENAI_API_KEY = 'sk-6869a1bf0b2f449f95b23627db6e965a'

client = openai.OpenAI(
    api_key = OPENAI_API_KEY,
    base_url = 'https://api.deepseek.com'
)

try:
    response = client.chat.completions.create(
        model='deepseek-chat',
        messages=[{'role': 'user', 'content': "test"}],
        max_tokens=1
    )
    print("api设置成功")
except Exception as e:
    print(f'api设置失败：{e}')
    
# TODO: 在此处输入用于摘要的提示词
prompt_for_meet = "你是一个人工智能专家，请面试我相关问题"
prompt_for_chatbot = "面试官"
# 重置对话的函数
def reset() -> List:
    return []

# 调用模型生成摘要的函数
def interact_roleplay(chatbot: List[Tuple[str, str]], userInput: str, temp=1.0) -> List[Tuple[str, str]]:
    messages = []
    for input, output in chatbot:
        messages.append({'role': 'user', 'content': input})
        messages.append({'role': 'assistant', 'content': output})
    messages.append({'role': 'user', 'content': userInput})
    response = client.chat.completions.create(
        model="deepseek-chat",  # 使用deepseek-v3
        messages=messages,   # 字典类型，每个元素必须要有角色和内容，角色可以有system，user，assistant，function
        temperature=temp,
        # max_tokens=200,  # 你需要注意到这里可以设置文本的长度上限，以节省token（测试时）
    )
    chatbot.append((userInput, response.choices[0].message.content))
    
    return chatbot

# 导出整个对话内容的函数
def export_summarization(chatbot: List[Tuple[str, str]], description: str) -> None:
    
    target = {"chatbot": chatbot, "description": description}
    with open("D:\\WorkSpace\\Python\\LLMStudy\\ApiCall\\files\\part2.json", "w") as file:
        json.dump(target, file)

first_dialogue = interact_roleplay([], prompt_for_meet)

# 生成 Gradio 的UI界面
with gr.Blocks() as demo:
    gr.Markdown("# 第1部分：摘要\n我正在面试你，请不要作弊！")
    chatbot = gr.Chatbot(value=first_dialogue)  # chatbot 中保存着对话记录，所以每次对话后需要更新chatbot
    description_textbox = gr.Textbox(label="机器人扮演的角色", interactive=False, value=f"{prompt_for_chatbot}")
    input_textbox = gr.Textbox(label="回答", interactive=True, value="填充")
    
    with gr.Column():
        gr.Markdown("# 温度调节\n温度用于控制聊天机器人的输出。温度越高，响应越具创造性。")
        temperature_slider = gr.Slider(0.0, 2.0, 1.0, step=0.1, label="温度")
    
    with gr.Row():
        sent_button = gr.Button(value="发送")
        reset_button = gr.Button(value="重置")

    with gr.Column():
        gr.Markdown("# 保存结果\n当你对结果满意后，点击导出按钮保存结果。")
        export_button = gr.Button(value="导出")
    
    # 连接按钮与函数
    sent_button.click(interact_roleplay, inputs=[chatbot, input_textbox, temperature_slider], outputs=[chatbot])
    reset_button.click(reset, outputs=[chatbot])
    export_button.click(export_summarization, inputs=[chatbot, input_textbox])

# 启动 Gradio 界面
demo.launch(debug=True)




