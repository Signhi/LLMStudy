import tqdm
import time
import json
import openai
import gradio as gr
import tiktoken
import os
import numpy as np
import pickle
import traceback
import jinja2
import re

max_prompt_token_num = 1024
os.chdir('D:\\WorkSpace\\Python\\LLMStudy\\prompt\\')
questions = np.loadtxt('./data/question.txt', dtype=str, encoding='UTF-8').tolist()
answers = np.loadtxt('./data/answers.txt', dtype=str).tolist()

OPENAI_API_KEY = 'sk-6869a1bf0b2f449f95b23627db6e965a'
base_url = 'https://api.deepseek.com'
client = openai.OpenAI(
    api_key= OPENAI_API_KEY,
    base_url = base_url
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
    
class OpenAIModel():
    def __init__(self, cache_file="openai_cache"):
        # 初始化 OpenAI 模型对象，并设置缓存文件
        self.cache_file = cache_file
        self.cache_dict = self.load_cache()  # 加载缓存

    def save_cache(self):
        # 将当前缓存保存到文件
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.cache_dict, f)

    def load_cache(self, allow_retry=True):
        # 从文件加载缓存，带有重试机制
        if os.path.exists(self.cache_file):
            while True:
                try:
                    with open(self.cache_file, "rb") as f:
                        cache = pickle.load(f)
                    break
                except Exception:
                    if not allow_retry:
                        assert False
                    print("Pickle Error: 5秒后重试...")
                    time.sleep(5)
        else:
            # 如果文件不存在则初始化缓存
            cache = {}
        return cache

    def set_cache_file(self, file_name):
        # 设置缓存文件名并加载缓存
        self.cache_file = file_name
        self.cache_dict = self.load_cache()

    def get_completion(self, content):
        # 获取模型完成的文本，先检查缓存，若无则请求生成
        # 如果选择检查缓存，则会导致同问题不同trial的结果相同，这与实际想表达的内容不符，故注释
        # if content in self.cache_dict:
        #     return self.cache_dict[content]
        for _ in range(3):
            try:
                # 调用模型生成内容
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{'role': 'user', 'content': content}],
                    temperature=1.0,
                )
                completion = response.choices[0].message.content
                self.cache_dict[content] = completion
                return completion
            except Exception as e:
                print(e, "\n")
                time.sleep(1)
        return None

    def is_valid_key(self):
        # 检查 API 密钥是否有效
        for _ in range(4):
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{'role': 'user', 'content': "hi there"}],
                    temperature=1.0,
                    max_tokens=1
                )
                return True
            except Exception as e:
                traceback.print_exc()
                time.sleep(1)
        return False

    def prompt_token_num(self, prompt):
        # 使用 tiktoken 来计算 token 数量
        try:
            # 使用 gpt-3.5-turbo 的编码器，因为 tiktoken 库不支持自动识别 qwen-turbo 模型
            encoding = tiktoken.get_encoding("cl100k_base")  # 这是 GPT-3.5-turbo 所使用的编码器
            # 将 prompt 编码成 token，并返回 token 数量
            tokens = encoding.encode(prompt)
            return len(tokens)
        except Exception as e:
            print(f"计算 token 数量时出错: {e}")
            return 0

    def two_stage_completion(self, question, content):
        # 两阶段完成：首先获取推理，再获取最终答案
        rationale = self.get_completion(content)
        if not rationale:
            return {
                'prompt': content,
                'rationale': None,
                'answer': None
            }

        ans = self.get_completion(content=f"Q:{question}\nA:{rationale}\nThe answer to the original question is (a number only): ")
        return {
            'prompt': content,
            'rationale': rationale,
            'answer': ans
        }



def Reset(chatbot):
    gr.Info('提示词已清除')

def Assign(chatbot, prompt, template, Example_Number):
    gr.Info('正在设置提示词')
    
    prompt_token_num = my_model.prompt_token_num(prompt)
    if prompt_token_num > max_prompt_token_num:
        template =None
        gr.Warning('提示词过长，请重新输入，提示词长度不能超过1024个token')
        return chatbot, prompt, template, Example_Number, prompt_token_num
    if Example_Number < 0 or Example_Number >= len(questions):
        template = None
        gr.Warning('questionID 不存在，请重新输入')
        return chatbot, prompt, template, Example_Number, prompt_token_num
    if "{{question}}" not in prompt:
        template = None
        gr.Warning('模板中不包含 {{question}} 变量，请重新输入')
        return chatbot, prompt, template, Example_Number, prompt_token_num
    environment = jinja2.Environment()
    template = environment.from_string(prompt)
    prompt_ex = f"""{template.render(question=questions[Example_Number - 1])}"""
    chatbot.extend([["分配提示词", "提示词已成功分配\n\n自定义提示词示例："], [None, prompt_ex]])
    return chatbot, prompt, template, Example_Number, prompt_token_num

def CleanCommas(text):
    pattern = r'\d{1:3}(?:,\d{3})*(?:.\d+)?'
    def ProcessCommas(match):
        number = match.group(0)  # 获取匹配到的数字字符串
        if '.' in number:
            return number
        number_list = number.split(',')
        new_number = number_list[0]
        for i in range(1, len(number_list)):
            if len(number_list[i]) == 3:
                    new_string += number_list[i]
            else:
                new_string += f",{number_list[i]}"
        return new_number
    return re.sub(pattern, ProcessCommas, text)  # 使用t替换后的字符串代替匹配到的字符串

def find_and_match_floats(input_string, ground_truth):
    pattern = re.compile('[-+]?\d*\.\d+|[-+]?\d+')  # | 表示匹配符号前和符号后的任意一个
    found_nums = pattern.findall(input_string)
    found_floats_nums = [float(num) for num in found_nums]
    return ground_truth in found_nums

def access(chatbot, template, test_num):
    if template == None:
        chatbot.extend([["提示词分配", "请先分配提示词"]])
        gr.Warning('未设置提示词')
        return chatbot, [], "提示词未设置", gr.Slider(label="Result Number", value=0, minimum=0, maximum=0, step=1), gr.Textbox(label="Result", value="", interactive=False)
    gr.Info('正在评估提示词')
    ans_template = "提示词和问题：\n\n{{question}}\n\n--------------------\n\n解题过程：\n\n{{rationale}}\n\n--------------------\n\n最终答案\n\n{{answer}}"
    res_list = []
    total_count = test_num
    environment = jinja2.Environment()
    ans_template = environment.from_string(ans_template)
    trial_num = 3
    trials = [[] for _ in range(trial_num)]  # 初始化一个二维空列表
    res_stats_str = ""
    
    for ii in range(trial_num):
        gr.Info(f'第{ii + 1}次尝试')
        accurate_count = 0
        for idx, example in enumerate(questions[:test_num]):
            test_res = ''
            result = my_model.two_stage_completion(example, template.render(question=example))

            if not result['answer']:
                trials.append(0)
                test_res += f'第{ii + 1}次尝试\n \n跳过Q{idx + 1}'
                test_res += '\n' + '='*10 + '\n'
                res_list.append(test_res)
                continue
            clean_res = CleanCommas(result['answer'])
            if find_and_match_floats(clean_res, answers[idx]):
                accurate_count += 1
                trials[ii].append(1)
            else:
                trials[ii].append(0)
                
            test_res += f'第{ii + 1}次尝试\n Q{idx + 1}' + '-' * 20 
            test_res += f'{ans_template.render(questions = result['prompt'], rationale=result['rationale'], answer=result['answer'])}\n'
            test_res += '\n' + '='*10 + '\n'
            res_list.append(test_res)
        res_stats_str = f'第{ii + 1}次尝试，正确题目数量为:{accurate_count}, 题目总数为:{total_count}, 正确率为:{accurate_count/total_count}\n'
        my_model.save_cache()
        
    voting_acc = 0
    for i in range(total_count):
        count = 0
        for j in range(trial_num):
            if trials[j][i] == 1:
                count += 1
        if count >= trial_num / 2:
            voting_acc += 1
    res_stats_str += f'投票法正确题目数量为:{voting_acc}, 题目总数为:{total_count}, 正确率为:{voting_acc/total_count}\n'
    chatbot.extend([["测试", "测试完成。结果可以在“结果”和“结果统计”中找到。"]])
    chatbot.extend([[None, "测试结果"], [None, ''.join(res_list)], [None, "结果统计"], [None, res_stats_str]])
    return chatbot, res_list, res_stats_str, gr.Slider(label="Result Number", value=1, minimum=1, maximum=len(res_list), step=1, visible=False), gr.Textbox(label="Result", value=res_list[0], interactive=False)

def SavePrompt(chatbot, prompt):
    gr.Info('正在保存提示词')
    prompt_dict = {
        'prompt': prompt
    }
    with open('./data/prompt.json', 'w') as f:
        json.dump(prompt_dict, f)
    chatbot.extend([["保存提示词", "提示词已保存为prompt.json"]])
    return chatbot, prompt
            
            
# 初始化模型
my_model = OpenAIModel()
with gr.Blocks() as demo:
    my_magic_prompt = "任务：\n解决以下数学问题。\n\n问题：{{question}}\n\n答案："
    my_magic_prompt = my_magic_prompt.strip('\n')
    template = gr.State(None)
    res_list = gr.State(list())

    # 组件
    with gr.Tab(label="Console"):
        with gr.Group():
            example_num_box = gr.Dropdown(label="Demo Example (Please choose one example for demo)", value=1, info=questions[0], choices=[i+1 for i in range(len(questions))], filterable=False)
            prompt_textbox = gr.Textbox(label="Custom Prompt", placeholder=f"在这里输入你的自定义提示词。例如：\n\n{my_magic_prompt}", value="", info="请确保包含`{{question}}`标签。")
            with gr.Row():
                set_button = gr.Button(value="Set Prompt")
                reset_button = gr.Button(value="Clear Prompt")
            prompt_token_num = gr.Textbox(label="Number of prompt tokens", value=0, interactive=False, info="自定义提示词的Token数量。")
        with gr.Group():
            test_num = gr.Slider(label="Number of examples used for evaluation", minimum=1, maximum=len(questions), step=1, value=1)
            assess_button = gr.Button(value="Evaluate")
        with gr.Group():
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        trial_no = gr.Slider(label="Trial ID", value=1, minimum=1, maximum=3, step=1)
                        ques_no = gr.Slider(label="Question ID", value=1, minimum=1, maximum=1, step=1)
                    res_num = gr.Slider(label="Result Number", value=0, minimum=0, maximum=0, step=1, visible=False)
                    res = gr.Textbox(label="Result", value="", placeholder="暂无结果", interactive=False)
                with gr.Column():
                    res_stats = gr.Textbox(label="Result Stats", interactive=False)
            save_button = gr.Button(value="Save Custom Prompt")
    with gr.Tab(label="Log"):
        chatbot = gr.Chatbot(label="Log")

    # 事件处理
    example_num_box.input(lambda Example_Number: gr.Dropdown(label="Example (Please choose one example for demo)", value=Example_Number, info=questions[Example_Number - 1], choices=[i+1 for i in range(len(questions))]),
                inputs=[example_num_box],
                outputs=[example_num_box])
    res_num.change(lambda results, result_num, test_num: (gr.Textbox(label="Result", value=results[result_num-1], interactive=False) if len(results) != 0 else gr.Textbox(label="Result", value="", placeholder="暂无结果", interactive=False),
                                    (int)((result_num-1)/test_num)+1,
                                    gr.Slider(label="Question Number", minimum=1, maximum=test_num, value=(result_num-1)%test_num+1, step=1)),
            inputs=[res_list, res_num, test_num],
            outputs=[res, trial_no, ques_no])
    trial_ques_no_input = lambda t_val, q_val, test_num: (t_val - 1) * test_num + q_val
    trial_no.input(trial_ques_no_input, inputs=[trial_no, ques_no, test_num], outputs=[res_num])
    ques_no.input(trial_ques_no_input, inputs=[trial_no, ques_no, test_num], outputs=[res_num])
    set_button.click(Assign, inputs=[chatbot, prompt_textbox, template, example_num_box], outputs=[chatbot, prompt_textbox, template, example_num_box, prompt_token_num])
    reset_button.click(Reset, inputs=[chatbot], outputs=[chatbot, prompt_textbox, prompt_token_num])
    assess_button.click(access, inputs=[chatbot, template, test_num], outputs=[chatbot, res_list, res_stats, res_num, res])
    save_button.click(SavePrompt, inputs=[chatbot, prompt_textbox], outputs=[chatbot])

demo.queue().launch(debug=True)
