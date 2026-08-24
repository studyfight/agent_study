import json
import os
from pathlib import Path
from typing import List, Dict, Any
import re # Added for get_step_number_from_content

import gradio as gr
from bs4 import BeautifulSoup

# 复用 app.py 中的文档与模型调用
from app import DOC_HTML, STEP_IMG_DIRS, call_llm  # type: ignore

BASE_DIR = Path(__file__).parent.resolve()


def collect_all_images() -> List[str]:
    """从 DOC_HTML 解析图片 src，映射为本地路径（供 Gradio 展示）。"""
    soup = BeautifulSoup(DOC_HTML, "html.parser")
    results: List[str] = []
    for img in soup.find_all("img"):
        src = img.get("src") or ""
        # 修复：使用正确的路径格式，与app.py中的路径一致
        if src.startswith("/static/guide_images/"):
            # 转换为本地文件路径
            local_path = BASE_DIR / "static" / src.replace("/static/", "")
            if local_path.exists():
                results.append(str(local_path))
    return results


# 直接从文档HTML解析1-8步（更稳定）

def extract_steps_from_doc_html() -> List[Dict[str, Any]]:
    """解析 DOC_HTML 文本，抽取编号为 1-8 的步骤，避免依赖 LLM。"""
    soup = BeautifulSoup(DOC_HTML, "html.parser")
    raw_text = soup.get_text("\n")
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    steps: List[Dict[str, Any]] = []
    current_num: Any = None
    buffer: List[str] = []

    def flush_current():
        nonlocal steps, current_num, buffer
        if current_num is None:
            return
        content_text = "\n".join(buffer).strip()
        if content_text:
            steps.append({
                "title": f"第{current_num}步",
                "content": f"{current_num}. {content_text}",
                "step_number": int(current_num),
            })
        buffer = []

    for ln in lines:
        m = re.match(r'^(\d)\.', ln)
        if m:
            # 新的步骤开始
            if current_num is not None:
                flush_current()
            current_num = int(m.group(1))
            rest = ln[m.end():].strip()
            buffer = [rest] if rest else []
        else:
            if current_num is not None:
                buffer.append(ln)

    flush_current()

    # 仅保留 1..8 步
    steps = [s for s in steps if 1 <= s.get("step_number", 0) <= 8]
    return steps


def plan_steps_with_llm(question: str) -> List[Dict[str, Any]]:
    """让模型输出连续的故障排查指导，然后分割成步骤"""
    # 增加长度限制，避免截断内容
    context = DOC_HTML[:20000]
    prompt = (
        "你是机械维修助手。请严格按照原始文档的步骤内容输出故障排查指导。\n\n"
        "要求：\n"
        "- 不要输出JSON格式\n"
        "- 用连续的段落描述故障排查步骤\n"
        "- 每个段落用数字编号（1. 2. 3. 4. 5. 6. 7. 8.）\n"
        "- 严格按照原始文档的步骤内容，不要修改或简化\n"
        "- 每个步骤最后写\"转下一步\"或\"→亮灯后转下一步\"\n"
        "- 第1步：检查开关是否在凿岩档位，凿岩灯亮灯，扳至凿岩档\n"
        "- 第2步：检查机械手是否在钎仓一侧，转钎锁接近开关是否亮灯；调整距离、检查线路，检查K7转钎锁继电器或更换继电器或接近开关\n"
        "- 第3步：按住开爪按钮，检查显示器上转钎锁灯是否亮灯，检查K7/K6继电器触点、底座、机连接线，保证019号线对地连通，必要时更换继电器\n"
        "- 第4步：按住开爪按钮，正转/反转手柄，检查显示器上的正转灯是否亮灯；手柄插头插拔检查是否接触不良或进水，在配电柜用万用表检查021/022号线是否有24V\n"
        "- 第5步：按住开爪按钮，正转/反转手柄，用万用表检测显示器输出端27/28号线有无24V电，没有24V需更换显示器\n"
        "- 第6步：打开推进梁上换钎电磁阀组盖板，按住开爪按钮，正转/反转手柄，检查转钎电磁阀插头是否亮灯\n"
        "- 第7步：启动机器，打到凿岩档、换钎档，用细的工具比如小螺丝刀或内六角顶换向阀芯，如卡死则更换换向阀\n"
        "- 第8步：拆下钎仓上的转钎马达和刹车总成，油管不拆，重复第6步骤，检查马达输出轴是否正常旋转\n"
        "- 必须严格按照8个步骤输出，内容要与原始文档一致\n\n"
        "请基于以下资料与用户问题生成：\n"
        f"<资料HTML>\n{context}\n</资料HTML>\n"
        f"<用户问题>\n{question}\n</用户问题>\n"
    )
    
    text = call_llm(prompt)
    cleaned = text.strip()
    
    print(f"🔍 LLM原始返回: {text[:300]}...")
    
    # 将文本分割成步骤
    steps = []
    
    # 使用正则表达式更准确地分割步骤（兼容前缀标题符号，如#、##、###）
    import re
    step_pattern = r'(?m)^\s*#*\s*(\d+\.\s*[^(\n)]*(?:\n(?!\s*#*\s*\d+\.)[^\n]*)*)'
    step_matches = re.findall(step_pattern, cleaned)
    
    for i, step_content in enumerate(step_matches, 1):
        # 清理步骤内容，移除重复的部分
        cleaned_content = step_content.strip()
        
        # 如果内容包含重复的说明，只保留第一部分
        if '请确保' in cleaned_content and '如果' in cleaned_content:
            # 找到第一个句号的位置
            first_period = cleaned_content.find('。')
            if first_period > 0:
                cleaned_content = cleaned_content[:first_period + 1]
        
        # 移除多余的换行和空格
        cleaned_content = re.sub(r'\n+', '\n', cleaned_content)
        cleaned_content = re.sub(r' +', ' ', cleaned_content)
        
        steps.append({
            "title": f"第{i}步",
            "content": cleaned_content,
            "step_number": i
        })
    
    # 如果正则表达式没有找到步骤，使用原来的方法作为备用
    if not steps:
        paragraphs = cleaned.split('\n\n')
        current_step = ""
        step_number = 1
        
        for para in paragraphs:
            para = para.strip()
            if para:
                # 如果段落以数字开头，开始新步骤
                if para[0].isdigit() and '.' in para[:3]:
                    # 保存之前的步骤
                    if current_step:
                        steps.append({
                            "title": f"第{step_number}步",
                            "content": current_step.strip(),
                            "step_number": step_number
                        })
                        step_number += 1
                    current_step = para
                else:
                    # 继续添加到当前步骤
                    current_step += "\n\n" + para
        
        # 添加最后一个步骤
        if current_step:
            steps.append({
                "title": f"第{step_number}步",
                "content": current_step.strip(),
                "step_number": step_number
            })
    
    print(f"✅ 分割成 {len(steps)} 个步骤")
    return steps


def get_step_images(step_number: int) -> List[str]:
    """根据步骤编号获取对应的图片"""
    # 从app.py导入新的图片获取函数
    try:
        from app import get_step_images as app_get_step_images
        return app_get_step_images(step_number)
    except ImportError:
        print("⚠️ 无法导入app模块的图片获取函数，使用备用方案")
        return []


def get_step_number_from_content(step_content: str) -> int:
    """从步骤内容中提取步骤编号"""
    # 查找步骤编号（兼容如"1."、"### 1." 等格式）
    match = re.search(r'^\s*#*\s*(\d+)\.', step_content.strip())
    if match:
        return int(match.group(1))
    
    # 如果没有找到编号，根据内容关键词推断
    if "安全" in step_content or "注意事项" in step_content:
        return 2  # 安全注意事项匹配到第2步的图片（控制面板）
    elif "开关" in step_content or "档位" in step_content:
        return 2
    elif "机械手" in step_content or "接近开关" in step_content:
        return 3
    elif "继电器" in step_content or "线路" in step_content:
        return 4
    elif "手柄" in step_content or "显示器" in step_content:
        return 5
    elif "电磁阀" in step_content:
        return 6
    elif "换向阀" in step_content or "阀芯" in step_content:
        return 7
    elif "电机" in step_content or "制动器" in step_content:
        return 8
    else:
        return 1  # 默认返回第1步


def get_step_images_by_content(step_content: str) -> List[str]:
    """根据步骤内容获取对应的图片"""
    # 从内容中提取步骤编号
    step_number = get_step_number_from_content(step_content)
    print(f"🔍 步骤内容: {step_content[:100]}...")
    print(f"📍 推断步骤编号: {step_number}")
    
    # 获取该步骤的图片
    images = get_step_images(step_number)
    print(f"🖼️ 步骤{step_number}的图片: {[os.path.basename(img) for img in images]}")
    
    return images


def format_step_md(step: Dict[str, Any], idx: int, total: int, images: List[str]) -> str:
    """格式化单个步骤显示"""
    title = step.get("title", f"第{idx + 1}步")
    content = step.get("content", "")
    
    formatted_content = f"# {title}\n\n"
    formatted_content += f"**步骤进度**: {idx + 1}/{total}\n\n"
    formatted_content += "---\n\n"
    
    # 处理内容，改善换行
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if line:
            # 如果行以数字开头（如"1."），添加换行
            if line[0].isdigit() and '.' in line[:3]:
                formatted_content += f"\n{line}\n"
            else:
                formatted_content += f"{line}\n"
    
    formatted_content += "\n---\n\n"
    
    # 添加操作提示
    if idx == 0:
        formatted_content += "**操作提示**: 请仔细按照说明操作，如有疑问可查看参考图片。\n\n"
    elif idx == total - 1:
        formatted_content += "**最后步骤**: 如果问题仍未解决，建议联系专业工程师。\n\n"
    else:
        formatted_content += "**操作提示**: 请仔细按照说明操作，如有疑问可查看参考图片。\n\n"
    
    formatted_content += "如本步无法解决，请点击\"下一步\"。若已解决，请点击\"已解决\"。"
    
    return formatted_content


# --- Gradio 回调 ---

def start_fn(question: str):
    """开始诊断"""
    if not question.strip():
        return "请输入问题", [], [], 0
    
    print(f"🚀 开始诊断问题: {question}")
    
    # 生成步骤
    steps = plan_steps_with_llm(question)
    print(f"📊 生成了 {len(steps)} 个步骤")
    
    # 显示第一步
    cur_idx = 0
    cur_step = steps[cur_idx]
    
    # 获取第一步对应的图片
    cur_images = get_step_images_by_content(cur_step["content"])
    
    # 格式化显示
    cur_md = format_step_md(cur_step, cur_idx, len(steps), cur_images)
    
    print(f"📍 当前步骤: {cur_idx + 1}/{len(steps)}")
    print(f"🖼️ 当前图片数量: {len(cur_images)}")
    
    return cur_md, cur_images, steps, cur_idx


def next_fn(steps: List[Dict[str, Any]], idx: int):
    """下一步"""
    print(f"⏭️ 下一步: 当前索引 {idx}, 总步骤数 {len(steps) if steps else 0}")
    
    if not steps:
        return "暂无步骤，请重新开始。", [], steps, idx
    
    nxt = idx + 1
    if nxt >= len(steps):
        md = (
            "## 🎯 排查完成\n\n"
            "已到最后一步。若问题仍未解决，建议联系专业工程师进行进一步检查与维修。\n\n"
            "**总结**: 请回顾所有步骤，确认是否遗漏了某些检查点。"
        )
        return md, [], steps, idx
    
    # 显示下一步
    next_step = steps[nxt]
    
    # 获取所有图片
    all_imgs = collect_all_images()
    
    # 获取下一步对应的图片
    next_images = get_step_images_by_content(next_step["content"])
    
    # 格式化显示
    md = format_step_md(next_step, nxt, len(steps), next_images)
    
    print(f"📍 显示步骤 {nxt + 1}/{len(steps)}, 图片数量: {len(next_images)}")
    
    return md, next_images, steps, nxt


def solved_fn(steps: List[Dict[str, Any]], idx: int):
    """问题已解决"""
    return (
        "## ✅ 问题已解决\n\n"
        "感谢使用志高机械故障排查系统！\n\n"
        "**提示**: 请记录本次排查的关键步骤，以便将来参考。\n"
        "本次排查结束。"
    ), [], steps, idx


def reset_fn():
    """重置状态"""
    return "请重新输入问题并点击\"开始诊断\"。", [], [], 0


# --- 构建 UI ---

def build_demo():
    with gr.Blocks(css="""
        .document-content {
            background: #ffffff;
            padding: 25px;
            border-radius: 12px;
            margin: 20px 0;
            border: 1px solid #e1e5e9;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            line-height: 1.8;
            font-size: 14px;
        }
        .step-content h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .image-gallery {
            margin-top: 20px;
            border-radius: 8px;
            overflow: hidden;
        }
        .status-info {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border: 1px solid #dee2e6;
            color: #495057;
        }
        .progress-info {
            background: #e3f2fd;
            padding: 12px;
            border-radius: 6px;
            margin: 15px 0;
            border: 1px solid #bbdefb;
            color: #1976d2;
            font-weight: 500;
        }
    """) as demo:
        
        gr.Markdown("# 🔧 换钎系统故障排查")
        gr.Markdown("**专业、安全的机械故障排查指导**")
        
        with gr.Row():
            with gr.Column(scale=1):
                q = gr.Textbox(
                    label="问题描述", 
                    value="换钎系统低压报警，应如何排查？",
                    placeholder="请描述您遇到的机械故障问题...",
                    lines=3
                )
                
                # 控制按钮
                with gr.Row():
                    btn_start = gr.Button("开始诊断", variant="primary", size="lg")
                    btn_reset = gr.Button("重新开始", variant="secondary")
                
                # 步骤导航
                with gr.Row():
                    btn_next = gr.Button("下一步", variant="secondary")
                    btn_done = gr.Button("已解决", variant="success")
                
                # 状态信息
                status_info = gr.Markdown("**状态**: 等待开始诊断")
            
            with gr.Column(scale=2):
                # 步骤内容显示
                step_display = gr.HTML(label="当前步骤", elem_classes=["document-content"])
                
                # 图片展示
                image_gallery = gr.Gallery(
                    label="步骤参考图", 
                    columns=3, 
                    height=400,
                    show_label=True,
                    elem_classes=["image-gallery"]
                )
                
                # 步骤进度
                progress_info = gr.Markdown("**步骤进度**: 0/0", elem_classes=["progress-info"])
        
        # 隐藏状态
        st_steps = gr.State([])  # List[Dict]
        st_idx = gr.State(0)

        # 事件绑定
        btn_start.click(
            start_fn, 
            inputs=[q], 
            outputs=[step_display, image_gallery, st_steps, st_idx]
        ).then(
            lambda steps, idx: f"**步骤进度**: {idx + 1}/{len(steps) if steps else 0}",
            inputs=[st_steps, st_idx],
            outputs=[progress_info]
        ).then(
            lambda: "**状态**: 诊断进行中，请按照步骤操作",
            outputs=[status_info]
        )
        
        btn_next.click(
            next_fn, 
            inputs=[st_steps, st_idx], 
            outputs=[step_display, image_gallery, st_steps, st_idx]
        ).then(
            lambda steps, idx: f"**步骤进度**: {idx + 1}/{len(steps) if steps else 0}",
            inputs=[st_steps, st_idx],
            outputs=[progress_info]
        )
        
        btn_done.click(
            solved_fn, 
            inputs=[st_steps, st_idx], 
            outputs=[step_display, image_gallery, st_steps, st_idx]
        ).then(
            lambda: "**状态**: 问题已解决",
            outputs=[status_info]
        )
        
        btn_reset.click(
            reset_fn,
            outputs=[step_display, image_gallery, st_steps, st_idx]
        ).then(
            lambda: "**步骤进度**: 0/0",
            outputs=[progress_info]
        ).then(
            lambda: "**状态**: 等待开始诊断",
            outputs=[status_info]
        )
        
        # 回车发送
        q.submit(
            lambda x: (x, "**状态**: 问题已设置，请点击\"开始诊断\""),
            inputs=[q],
            outputs=[q, status_info]
        )
    
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue().launch(server_name="127.0.0.1", server_port=8013, share=False) 

# app.py
"""
换钎系统故障排查后端（FastAPI）
- 负责：
  1) 解析与落地文档图片到分步目录；
  2) 提供按步骤读取图片的能力；
  3) 兼容调用大模型（DashScope 兼容模式）；
  4) 暴露静态资源与健康检查接口。

目录说明：
- static/guide_images/stepX_*：按步骤保存的图片文件夹
- DOC_HTML：由DOCX转换得到的HTML文本，用于前端/Gradio侧展示与检索

本文件中注释均为中文，尽量描述“做什么/为什么做”。
"""
import os
import uuid
from pathlib import Path
from typing import List, Dict

import requests
from fastapi import FastAPI, Body, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup
import mammoth
from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# 路径与常量
# -----------------------------
BASE_DIR = Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "static"
IMG_DIR = STATIC_DIR / "guide_images"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

# 为每个步骤创建子文件夹
STEP_IMG_DIRS = {
    1: IMG_DIR / "step1_safety",      # 第1步：安全注意事项
    2: IMG_DIR / "step2_switch",      # 第2步：检查开关档位
    3: IMG_DIR / "step3_manipulator", # 第3步：检查机械手
    4: IMG_DIR / "step4_relay",       # 第4步：检查继电器
    5: IMG_DIR / "step5_display",     # 第5步：检查显示器
    6: IMG_DIR / "step6_valve",       # 第6步：检查电磁阀
    7: IMG_DIR / "step7_directional", # 第7步：检查换向阀
    8: IMG_DIR / "step8_motor"        # 第8步：检查电机
}

# 创建步骤文件夹
for step_dir in STEP_IMG_DIRS.values():
    step_dir.mkdir(parents=True, exist_ok=True)

# 文档默认路径（优先同目录下的 docx）
DOCX_CANDIDATES: List[Path] = [
    BASE_DIR / "换钎系统故障排查.docx",
]

# DashScope 兼容端点与模型（你已验证可用）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DASHSCOPE_MODEL = "deepseek-r1-distill-qwen-14b"

# -----------------------------
# 读取并转换 DOCX -> HTML（包含图片落地到对应步骤文件夹）
# -----------------------------

def _save_image(image, step_number: int = None):
    """
    保存DOCX中抽出的图片到对应步骤文件夹。

    设计要点：
    - 根据当前已写入图片总数(total_images)推断归属的步骤，严格对齐文档中图片分布；
    - 针对每个步骤限制最大图片数量，避免误分配；
    - 生成可读的文件前缀（如 04_手柄、06_电路图），便于排序与回溯。
    """
    content_type = getattr(image, "content_type", None) or "image/png"
    ext = content_type.split("/")[-1]
    
    # 按照文档的实际图片分布来分配
    # 第1步：1张图片（控制面板）
    # 第2步：2张图片（机械手 + 电路图）
    # 第3步：3张图片（手柄 + 显示器 + 电路图）
    # 第4步：3张图片（手柄 + 显示器 + 电路图）
    # 第5步：0张图片
    # 第6步：1张图片（电磁阀）
    # 第7步：1张图片（换向阀）
    # 第8步：0张图片
    # 总共：1+2+3+3+0+1+1+0 = 11张图片
    
    # 获取当前已存在的图片总数（所有步骤的总和）
    total_images = 0
    for step_dir in STEP_IMG_DIRS.values():
        if step_dir.exists():
            for ext_pattern in ['*.png', '*.jpg', '*.jpeg', '*.gif']:
                total_images += len(list(step_dir.glob(ext_pattern)))
    
    # 根据图片总数，按照文档顺序分配
    if total_images == 0:
        step_number = 1  # 第1张图片 -> 第1步
    elif total_images == 1:
        step_number = 2  # 第2张图片 -> 第2步
    elif total_images == 2:
        step_number = 2  # 第3张图片 -> 第2步（第2步有2张图片）
    elif total_images == 3:
        step_number = 3  # 第4张图片 -> 第3步
    elif total_images == 4:
        step_number = 3  # 第5张图片 -> 第3步（第3步有3张图片）
    elif total_images == 5:
        step_number = 3  # 第6张图片 -> 第3步（第3步有3张图片）
    elif total_images == 6:
        step_number = 4  # 第7张图片 -> 第4步
    elif total_images == 7:
        step_number = 4  # 第8张图片 -> 第4步（第4步有3张图片）
    elif total_images == 8:
        step_number = 4  # 第9张图片 -> 第4步（第4步有3张图片）
    elif total_images == 9:
        step_number = 6  # 第10张图片 -> 第6步（跳过第5步，第5步无图片）
    elif total_images == 10:
        step_number = 7  # 第11张图片 -> 第7步
    else:
        # 如果图片数量超过预期，停止分配
        print(f"⚠️ 图片数量已超过预期（{total_images}张），停止分配")
        return {"src": ""}
    
    # 检查当前步骤是否还能容纳更多图片
    current_step_images = len(list(STEP_IMG_DIRS[step_number].glob('*.png'))) + \
                         len(list(STEP_IMG_DIRS[step_number].glob('*.jpg'))) + \
                         len(list(STEP_IMG_DIRS[step_number].glob('*.jpeg'))) + \
                         len(list(STEP_IMG_DIRS[step_number].glob('*.gif')))
    
    # 每个步骤的最大图片数量
    max_images_per_step = {
        1: 1,  # 第1步：1张
        2: 2,  # 第2步：2张
        3: 3,  # 第3步：3张
        4: 3,  # 第4步：3张
        5: 0,  # 第5步：0张
        6: 1,  # 第6步：1张
        7: 1,  # 第7步：1张
        8: 0   # 第8步：0张
    }
    
    if current_step_images >= max_images_per_step[step_number]:
        print(f"⚠️ 步骤{step_number}已达到最大图片数量（{max_images_per_step[step_number]}张），跳过")
        return {"src": ""}
    
    # 生成有意义的文件名，便于排序
    step_name = f"step{step_number}"
    
    # 根据步骤和当前图片数量，添加有意义的标识
    if step_number == 1:
        # 第1步：控制面板
        prefix = "01_控制面板"
    elif step_number == 2:
        if current_step_images == 0:
            prefix = "02_机械手"
        else:
            prefix = "03_电路图"
    elif step_number == 3:
        if current_step_images == 0:
            prefix = "04_手柄"
        elif current_step_images == 1:
            prefix = "05_显示器"
        else:
            prefix = "06_电路图"
    elif step_number == 4:
        if current_step_images == 0:
            prefix = "07_手柄"
        elif current_step_images == 1:
            prefix = "08_显示器"
        else:
            prefix = "09_电路图"
    elif step_number == 6:
        prefix = "10_电磁阀"
    elif step_number == 7:
        prefix = "11_换向阀"
    else:
        prefix = f"{current_step_images + 1:02d}_图片"
    
    name = f"{prefix}_{uuid.uuid4().hex[:8]}.{ext}"
    out_path = STEP_IMG_DIRS[step_number] / name
    
    with image.open() as img_in, open(out_path, "wb") as img_out:
        img_out.write(img_in.read())
    
    print(f"📸 第{total_images + 1}张图片 {name} 保存到 {step_name} 文件夹（当前步骤第{current_step_images + 1}张）")
    
    # mammoth 期望返回字典格式
    return {"src": f"/static/guide_images/{step_name}/{name}"}


def _load_doc_html() -> str:
    for cand in DOCX_CANDIDATES:
        if cand.exists():
            with open(cand, "rb") as f:
                result = mammoth.convert_to_html(
                    f, convert_image=mammoth.images.inline(_save_image)
                )
            return result.value or ""
    return ""


DOC_HTML = _load_doc_html()

# -----------------------------
# 获取步骤图片的函数
# -----------------------------

def get_step_images(step_number: int) -> List[str]:
    """获取指定步骤的图片列表，按照文档中的正确顺序"""
    if step_number not in STEP_IMG_DIRS:
        return []
    
    step_dir = STEP_IMG_DIRS[step_number]
    if not step_dir.exists():
        return []
    
    # 获取该步骤文件夹下的所有图片
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.gif']:
        image_files.extend(step_dir.glob(ext))
    
    # 按照文件名前缀排序，确保图片按正确顺序显示
    def get_order_key(filename):
        name = filename.name
        # 提取文件名前缀的数字部分进行排序
        if '_' in name:
            prefix = name.split('_')[0]
            try:
                return int(prefix)
            except ValueError:
                return 999  # 无法解析的数字放在最后
        return 999
    
    image_files = sorted(image_files, key=get_order_key)
    
    # 转换为字符串路径
    return [str(img) for img in image_files]


def get_all_step_images() -> Dict[int, List[str]]:
    """获取所有步骤的图片映射"""
    result = {}
    for step_num in STEP_IMG_DIRS.keys():
        result[step_num] = get_step_images(step_num)
    return result


def collect_all_images() -> List[str]:
    """从 DOC_HTML 解析图片 src，映射为本地路径（供 Gradio 展示）。"""
    soup = BeautifulSoup(DOC_HTML, "html.parser")
    results: List[str] = []
    for img in soup.find_all("img"):
        src = img.get("src") or ""
        # 新的路径格式：/static/guide_images/stepX_name/filename.ext
        if src.startswith("/static/guide_images/"):
            # 转换为本地文件路径
            local_path = BASE_DIR / "static" / src.replace("/static/", "")
            if local_path.exists():
                results.append(str(local_path))
    return results

# -----------------------------
# LLM 调用（DashScope 兼容）
# -----------------------------

def call_llm(prompt: str) -> str:
    """调用 DashScope 兼容的 chat.completions 接口，返回纯文本内容。"""
    if not DASHSCOPE_API_KEY:
        return "未配置 DASHSCOPE_API_KEY，返回占位答案。"
    payload = {
        "model": DASHSCOPE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是机械维修助手。请基于提供资料进行分步指导："
                    "1) 先给安全注意事项；2) 列出所需工具；3) 分步排查与操作；"
                    "4) 给出未解决时的进一步建议。不要做诊断性结论。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 512,
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(DASHSCOPE_URL, json=payload, headers=headers, timeout=60)
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=resp.text)
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


# -----------------------------
# FastAPI 应用
# -----------------------------
app = FastAPI(title="换钎系统故障排查 Demo", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health")
def health():
    return {
        "ok": True,
        "doc_loaded": bool(DOC_HTML),
        "images_dir": str(IMG_DIR),
    }


@app.post("/guide")
def guide(question: str = Body(..., embed=True)):
    if not DOC_HTML:
        raise HTTPException(status_code=500, detail="未找到或无法读取 DOCX 文档。")

    soup = BeautifulSoup(DOC_HTML, "html.parser")
    imgs = [img.get("src") for img in soup.find_all("img")][:6]

    context = DOC_HTML[:12000]
    prompt = (
        "基于以下资料给出分步指导，强调安全、工具、操作步骤与后续建议：\n"
        "<资料HTML>\n" + context + "\n</资料HTML>\n"
        "<用户问题>\n" + question + "\n</用户问题>"
    )
    answer = call_llm(prompt)
    return {"answer": answer, "images": imgs}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8010, reload=True)
