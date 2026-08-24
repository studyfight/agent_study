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
# 新增：多模态（视觉）模型名（需具备图像理解能力），不再给默认值，未配置则视为不可用
DASHSCOPE_VL_MODEL = os.getenv("DASHSCOPE_VL_MODEL", "")

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

# 新增：调用多模态模型进行图片相关性二分类（YES/NO）
import base64

def is_vl_enabled() -> bool:
    """是否启用了多模态模型（需要 API_KEY 与 VL 模型名同时存在）。"""
    return bool(DASHSCOPE_API_KEY and DASHSCOPE_VL_MODEL)

def judge_image_relevance_with_llm(image_bytes: bytes) -> "None | bool":
    """使用 DashScope 的多模态模型判断图片是否与换钎/转钎系统相关。
    返回：True/False 表示可判定；None 表示不可判定（如未配置模型/调用失败）。"""
    if not is_vl_enabled():
        return None
    try:
        img_b64 = base64.b64encode(image_bytes).decode("ascii")
        # 严格 YES/NO 提示
        prompt_text = (
            "请你判断这张图片是否为‘换钎/转钎系统’相关设备部件（如：转钎锁、转钎马达、换向阀、电磁阀、手柄、显示器、控制面板、钎仓等）。\n"
            "如果是，请仅输出 YES；如果不是，请仅输出 NO。请不要输出其他内容。"
        )
        payload = {
            "model": DASHSCOPE_VL_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            "max_tokens": 10,
            "temperature": 0.0,
        }
        headers = {
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json",
        }
        resp = requests.post(DASHSCOPE_URL, json=payload, headers=headers, timeout=60)
        if resp.status_code >= 400:
            print(f"⚠️ 视觉模型调用失败: {resp.status_code} {resp.text}")
            return None
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        low = content.lower()
        if low == "yes" or low == "是":
            return True
        if low == "no" or low == "否":
            return False
        # 容错：包含关键词也判定
        if "yes" in low or "是" in content:
            return True
        if "no" in low or "否" in content:
            return False
        return None
    except Exception as e:
        print(f"⚠️ 视觉模型异常: {e}")
        return None


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
