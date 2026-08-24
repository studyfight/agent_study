from dotenv import load_dotenv
import os
from pathlib import Path

# 定位 .env 文件（现在在当前文件夹config内）
current_file = Path(__file__)  # 获取当前load_key.py的路径
config_folder = current_file.parent  # 得到config文件夹路径（就是当前文件夹）
env_file_path = config_folder / ".env"  # 拼接出.env文件路径（config/.env）

# 加载.env文件并检查
load_success = load_dotenv(dotenv_path=env_file_path)
if not load_success:
    raise FileNotFoundError(f"无法加载 .env 文件，请检查路径：{env_file_path}")

def load_key(key_name):
    key = os.getenv(key_name)
    if not key:
        raise ValueError(f"在 .env 文件中未找到 {key_name}，请检查变量名是否正确")
    return key
    