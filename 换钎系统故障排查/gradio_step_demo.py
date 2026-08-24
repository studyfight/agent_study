"""
换钎系统故障排查 - Gradio 步进式演示界面

职责：
- 接收用户问题，调用后端的文档与图片资源
- 通过LLM（或固定模板）生成分步排查文本
- 每一步按编号加载相应图片，支持“下一步/已解决/重新开始”

结构：
- start_fn/next_fn/solved_fn/reset_fn：UI事件回调
- plan_steps_with_llm：生成分步文本
- get_step_images_by_content：根据文本推断编号并取图
- format_step_md：拼接Markdown用于HTML展示

本模块注释以中文为主，强调参数/返回值和设计决策。
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import re # Added for get_step_number_from_content

import gradio as gr
from bs4 import BeautifulSoup

# 复用 app.py 中的文档与模型调用
from app import DOC_HTML, STEP_IMG_DIRS, call_llm  # type: ignore
# 新增：尝试导入视觉判断函数
try:
	from app import judge_image_relevance_with_llm, is_vl_enabled  # type: ignore
except Exception:
	judge_image_relevance_with_llm = None  # type: ignore
	def is_vl_enabled() -> bool:  # type: ignore
		return False

BASE_DIR = Path(__file__).parent.resolve()

# 新增：与换钎/转钎系统相关的关键词与辅助函数
RELEVANT_KEYWORDS = [
    "换钎", "转钎", "换钎系统", "转钎系统", "转钎锁", "转钎马达", "转钎电机",
    "钎仓", "换向阀", "电磁阀", "凿岩档", "换钎档", "开爪", "推进梁", "刹车"
]

def is_relevant_question(question: str) -> bool:
    """判断问题是否与换钎/转钎系统相关。"""
    q = (question or "").lower()
    # 中文大小写无关，这里统一 lower 仅对英文字母生效
    return any(kw in q for kw in RELEVANT_KEYWORDS)


def refuse_with_llm(question: str) -> str:
    """调用大模型生成礼貌拒答，说明仅支持换钎/转钎相关问题。"""
    try:
        prompt = (
            "你是机械维修系统的前台助手。本系统仅支持‘换钎/转钎’相关的故障排查。\n"
            "对于与换钎系统无关的问题，请礼貌简短地拒绝，并提示用户将问题限定为换钎/转钎相关。\n"
            "请用中文输出，一段话即可，口吻专业、友好。\n\n"
            f"<用户问题>\n{question}\n</用户问题>\n"
        )
        reply = call_llm(prompt).strip()
        if not reply:
            raise ValueError("empty reply")
        return reply
    except Exception:
        return "抱歉，本系统仅支持换钎/转钎相关的故障排查。请将问题限定为换钎系统相关后再试。"

# 新增：图片相关性判定（基于简单感知哈希）
try:
    from PIL import Image
except ImportError:  # 兼容环境缺少 PIL 的情况
    Image = None  # type: ignore

from functools import lru_cache

def _average_hash(img: "Image.Image", hash_size: int = 8) -> int:
    """计算图像的平均哈希（aHash），返回整数位图。"""
    if Image is None:
        return -1
    # 转灰度并缩放至 hash_size x hash_size
    small = img.convert("L").resize((hash_size, hash_size))
    pixels = list(small.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for i, p in enumerate(pixels):
        if p >= avg:
            bits |= (1 << i)
    return bits

def _hamming_distance(a: int, b: int) -> int:
    if a < 0 or b < 0:
        return 1_000_000
    x = a ^ b
    dist = 0
    while x:
        dist += 1
        x &= x - 1
    return dist

# 新增：dHash（差值哈希）

def _difference_hash(img: "Image.Image", hash_size: int = 8) -> int:
    if Image is None:
        return -1
    # 生成 (hash_size+1) x hash_size，以计算相邻差值
    small = img.convert("L").resize((hash_size + 1, hash_size))
    pixels = list(small.getdata())
    bits = 0
    idx = 0
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[row_start + col]
            right = pixels[row_start + col + 1]
            if left < right:
                bits |= (1 << idx)
            idx += 1
    return bits

# 新增：汇总参考图片路径（HTML解析 + STEP_IMG_DIRS 目录遍历）
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

def gather_step_image_paths() -> List[str]:
    paths: List[str] = []
    # 来自 HTML 的图片
    paths.extend(collect_all_images())
    # 来自 STEP_IMG_DIRS 的图片（兼容 dict/list/single path）
    try:
        cands: List[Path] = []
        if isinstance(STEP_IMG_DIRS, dict):
            for v in STEP_IMG_DIRS.values():
                if v:
                    cands.append(Path(v))
        elif isinstance(STEP_IMG_DIRS, (list, tuple)):
            for v in STEP_IMG_DIRS:
                if v:
                    cands.append(Path(v))
        else:
            if STEP_IMG_DIRS:
                cands.append(Path(STEP_IMG_DIRS))
        for d in cands:
            if not d:
                continue
            dir_path = Path(d)
            if not dir_path.exists():
                continue
            for p in dir_path.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    paths.append(str(p))
    except Exception as e:
        print(f"⚠️ 收集参考图异常: {e}")
    # 去重
    seen = set()
    unique = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique

@lru_cache(maxsize=1)
def _reference_hashes_dual() -> List[Any]:
    """返回参考图的 (aHash, dHash) 列表。"""
    pairs: List[Any] = []
    if Image is None:
        return pairs
    for path in gather_step_image_paths():
        try:
            with Image.open(path) as im:
                a = _average_hash(im)
                d = _difference_hash(im)
                pairs.append((a, d))
        except Exception:
            continue
    return pairs

# 兼容旧接口，保留原有 _reference_hashes
@lru_cache(maxsize=1)
def _reference_hashes() -> List[int]:
    hashes: List[int] = []
    for a, _ in _reference_hashes_dual():
        hashes.append(a)
    return hashes

# 新增：返回（是否相关, 是否能够判断）

def assess_image_relevance(uploaded: Any, threshold: int = 12) -> (bool, bool):
	"""评估上传图片是否与换钎系统相关。返回二元组：(is_relevant, can_judge)。
	- 当 PIL 不可用或参考图为空、或无法打开图片时，返回 (False, False)。
	- 正常对比时，只要 aHash 或 dHash 与参考任一图的距离 <= 阈值，则判定为相关。
	"""
	if Image is None or uploaded is None:
		print("🔎 assess: PIL/上传为空 -> cannot judge")
		return False, False
	try:
		if isinstance(uploaded, Image.Image):
			up_img = uploaded
		else:
			up_img = Image.open(uploaded)
		refs = _reference_hashes_dual()
		print(f"🔎 assess: refs={len(refs)}")
		if not refs:
			# 哈希无参考，尝试 LLM 视觉判断
			if judge_image_relevance_with_llm is not None and is_vl_enabled():
				import io
				buf = io.BytesIO()
				up_img.convert("RGB").save(buf, format="JPEG", quality=85)
				res = judge_image_relevance_with_llm(buf.getvalue())
				print(f"🔎 assess: llm_fallback={res}")
				if isinstance(res, bool):
					return res, True
			return False, False
		up_a = _average_hash(up_img)
		up_d = _difference_hash(up_img)
		# 计算最小距离
		min_a = min(_hamming_distance(up_a, a) for a, _ in refs)
		min_d = min(_hamming_distance(up_d, d) for _, d in refs)
		is_rel = (min_a <= threshold) or (min_d <= threshold)
		print(f"🧮 图片相似度：min_a={min_a}, min_d={min_d}, 阈值={threshold}")
		if is_rel:
			return True, True
		# 哈希判定为不相关，进一步用 LLM 视觉兜底（如可用）
		if judge_image_relevance_with_llm is not None and is_vl_enabled():
			import io
			buf = io.BytesIO()
			up_img.convert("RGB").save(buf, format="JPEG", quality=85)
			res = judge_image_relevance_with_llm(buf.getvalue())
			print(f"🔎 assess: llm_after_hash={res}")
			if isinstance(res, bool):
				return res, True
		return False, True
	except Exception as e:
		print(f"⚠️ 图片判定异常: {e}")
		# 异常时尝试 LLM 视觉兜底
		try:
			if judge_image_relevance_with_llm is not None and is_vl_enabled():
				import io
				if isinstance(uploaded, Image.Image):
					up_img = uploaded
				else:
					up_img = Image.open(uploaded)
				buf = io.BytesIO()
				up_img.convert("RGB").save(buf, format="JPEG", quality=85)
				res = judge_image_relevance_with_llm(buf.getvalue())
				print(f"🔎 assess: llm_on_exception={res}")
				if isinstance(res, bool):
					return res, True
		except Exception:
			pass
		return False, False

# 兼容旧接口，保留但改为调用 assess

def is_relevant_image(uploaded: Any, threshold: int = 24) -> bool:
    is_rel, can_judge = assess_image_relevance(uploaded, threshold)
    return is_rel if can_judge else False


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
    """
    基于用户问题与文档上下文生成分步排查文本。

    返回格式：[{title, content, step_number}]，不直接面向UI。
    注意：LLM输出可能不稳定，已在后续做了编号解析与兜底。
    """
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
    """
    从步骤内容中提取步骤编号；若解析不到，采用关键词推断。

    设计取舍：
    - 先解析显式编号（稳定且准确）；
    - 再用关键词映射到常识步骤（容错）。
    """
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
    """
    将步骤结构化数据渲染为HTML片段（Markdown为主）。

    - 添加“步骤进度/分隔线/提示条”等辅助信息；
    - 文本处理：对编号行做额外换行，增强可读性。
    """
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

def start_fn(question: str, uploaded_image: Any = None):
	"""开始诊断（文字必选，图片可选）。

	进入排查条件：

	- 未上传图片：仅需文字包含换钎/转钎关键词；
	- 已上传图片：文字需包含关键词，且图片需判定为换钎系统相关；

	任一不满足则直接给出对应提示并不进入排查界面。
	"""
	if not question.strip():
		return "请输入问题", [], [], 0

	# 避免参考图缓存过期：每次开始诊断前刷新参考哈希
	try:
		_reference_hashes_dual.cache_clear()  # type: ignore
		ref_cnt = len(_reference_hashes_dual())  # type: ignore
		print(f"🗂️ 参考图片数量: {ref_cnt}")
	except Exception as e:
		print(f"⚠️ 刷新参考图缓存失败: {e}")

	has_image = uploaded_image is not None
	text_ok = is_relevant_question(question)
	print(f"🔎 gate: has_image={has_image}, text_ok={text_ok}, vl_enabled={is_vl_enabled()}")

	# 未上传图片：仅校验文本
	if not has_image:
		if not text_ok:
			refuse_text = refuse_with_llm(question)
			md = (
				"## 🙏 暂不支持该问题\n\n"
				f"{refuse_text}\n\n"
				"请包含‘换钎/转钎’等关键词后再试。"
			)
			return md, [], [], 0
		# 否则进入流程
	else:
		# 已上传图片：需文本与图片同时满足
		is_rel, can_judge = assess_image_relevance(uploaded_image)
		print(f"🔎 gate: can_judge={can_judge}, is_rel={is_rel}")
		if not text_ok:
			refuse_text = refuse_with_llm(question)
			md = (
				"## 🙏 暂不支持该问题\n\n"
				f"{refuse_text}\n\n"
				"当上传图片时，文字也需包含‘换钎/转钎’等关键词。"
			)
			return md, [], [], 0
		# 若启用了多模态模型，优先用 LLM 严格判定
		if is_vl_enabled() and judge_image_relevance_with_llm is not None:
			try:
				import io
				if isinstance(uploaded_image, Image.Image):
					up_img = uploaded_image
				else:
					up_img = Image.open(uploaded_image)
				buf = io.BytesIO()
				up_img.convert("RGB").save(buf, format="JPEG", quality=85)
				llm_res = judge_image_relevance_with_llm(buf.getvalue())
				print(f"🔎 gate: llm_res={llm_res}")
			except Exception:
				llm_res = None
			if llm_res is not True:
				# LLM 认为不是，或无法判定，均不放行并给出明确提示
				if llm_res is False:
					md = (
						"## ❌ 图片错误\n\n"
						"检测到上传图片非换钎/转钎系统相关，请上传正确的设备部位图片后重试。\n\n"
						"可参考文档中的步骤参考图进行对照。"
					)
				else:
					md = (
						"## ❌ 图片无法判定\n\n"
						"当前无法确认上传图片是否为换钎/转钎系统相关，请上传更清晰或角度更合适的设备部位图片后重试。"
					)
				return md, [], [], 0
			# LLM 判断为相关时，再要求哈希也通过以降低误判
			if not (can_judge and is_rel):
				md = (
					"## ❌ 图片错误\n\n"
					"图片未通过本地一致性校验，请上传文档中对应部位的清晰图片后重试。"
				)
				return md, [], [], 0
		else:
			# 未启用多模态，仅依赖本地哈希，严格把关
			if not can_judge:
				md = (
					"## ❌ 图片无法判定\n\n"
					"当前无法确认上传图片是否为换钎/转钎系统相关，请上传更清晰或角度更合适的设备部位图片后重试。"
				)
				return md, [], [], 0
			if not is_rel:
				md = (
					"## ❌ 图片错误\n\n"
					"检测到上传图片非换钎/转钎系统相关，请上传正确的设备部位图片后重试。\n\n"
					"可参考文档中的步骤参考图进行对照。"
				)
				return md, [], [], 0

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
                gr.Markdown(
                    """
                    提示参考：请尽量包含“换钎/转钎”等关键词，或上传换钎系统相关部件照片。
                    示例：
                    - 转钎锁不亮灯，应该如何一步步排查？
                    - 换钎系统低压报警，按文档步骤怎么检查？
                    - 按住开爪+正反转后，电磁阀不亮灯怎么办？
                    """
                )
                img = gr.Image(
                    label="上传设备图片（可选）",
                    type="pil",
                    height=220,
                    sources=["upload"]
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
            inputs=[q, img], 
            outputs=[step_display, image_gallery, st_steps, st_idx]
        ).then(
            lambda steps, idx: f"**步骤进度**: {idx + 1}/{len(steps) if steps else 0}",
            inputs=[st_steps, st_idx],
            outputs=[progress_info]
        ).then(
            lambda steps: ("**状态**: 诊断进行中，请按照步骤操作" if steps else "**状态**: 非换钎相关或图片错误，已拒答"),
            inputs=[st_steps],
            outputs=[status_info]
        )

        # 输入变更提示：需重新点击“开始诊断”
        q.change(
            lambda: "**状态**: 输入已更新，请点击\"开始诊断\"重新校验",
            outputs=[status_info]
        )
        img.change(
            lambda: "**状态**: 输入已更新，请点击\"开始诊断\"重新校验",
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