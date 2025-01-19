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
prompt_for_summarization = "请将以下文章概括成几句话。"

# 重置对话的函数
def reset() -> List:
    return []

# 调用模型生成摘要的函数
def interact_summarization(prompt: str, article: str, temp=1.0) -> List[Tuple[str, str]]:
    '''
    参数:
      - prompt: 我们在此部分中使用的提示词
      - article: 需要摘要的文章
      - temp: 模型的温度参数。温度用于控制聊天机器人的输出。温度越高，响应越具创造性。
    '''
    input = f"{prompt}\n{article}"
    response = client.chat.completions.create(
        model="deepseek-chat",  # 使用deepseek-v3
        messages=[{'role': 'user', 'content': input}],
        temperature=temp,
        max_tokens=200,  # 你需要注意到这里可以设置文本的长度上限，以节省token（测试时）
    )

    return [(input, response.choices[0].message.content)]

# 导出整个对话内容的函数
def export_summarization(chatbot: List[Tuple[str, str]], article: str) -> None:
    '''
    参数:
      - chatbot: 模型的对话记录，存储在元组列表中
      - article: 需要摘要的文章
    '''
    target = {"chatbot": chatbot, "article": article}
    with open("D:\\WorkSpace\\Python\\LLMStudy\\ApiCall\\files\\part1.json", "w") as file:
        json.dump(target, file)

# 生成 Gradio 的UI界面
with gr.Blocks() as demo:
    gr.Markdown("# 第1部分：摘要\n填写任何你喜欢的文章，让聊天机器人为你总结！")
    chatbot = gr.Chatbot()
    prompt_textbox = gr.Textbox(label="提示词", value=prompt_for_summarization, visible=False)
    article_textbox = gr.Textbox(label="文章", interactive=True, value="填充")
    
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
    sent_button.click(interact_summarization, inputs=[prompt_textbox, article_textbox, temperature_slider], outputs=[chatbot])
    reset_button.click(reset, outputs=[chatbot])
    export_button.click(export_summarization, inputs=[chatbot, article_textbox])

# 启动 Gradio 界面
demo.launch(debug=True)





