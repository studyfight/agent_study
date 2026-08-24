import random

from Director import graph
#
# query = "给我讲一个郭德纲的笑话"
#
# config = {
#         "configurable":{
#             "thread_id":random.randint(1,10000)
#         }
#     }
#
# res = graph.invoke({"messages":["今天天气怎么样"]}
#                        ,config
#                        ,stream_mode="values")
# print(res["messages"][-1].content)

# grdio前端
import gradio as gr
def process_input(text):
    config = {
        "configurable":{
            "thread_id":random.randint(1,10000)
        }
    }

    result = graph.invoke({"messages":[text]},config)
    return result["messages"][-1].content

with gr.Blocks() as demo:
    gr.Markdown("# LangGraph Multi - Agent")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 可以问路线规划，对对联，讲笑话，快来试试吧。")
            inputs_text = gr.Textbox(label="问题*", placeholder="请输入你的问题", value="讲一个郭德纲的笑话")
            btn_start = gr.Button("开始",variant="primary")
        with gr.Column():
            output_text = gr.Textbox(label="回答")
    btn_start.click(process_input, inputs=[inputs_text], outputs=[output_text])
demo.launch()