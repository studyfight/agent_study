import os
from dotenv import load_dotenv
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_api_key():
    """
    从.env文件加载API密钥
    
    Returns:
        str: API密钥，如果加载失败返回None
    """
    try:
        # 加载.env文件
        load_dotenv()
        
        # 获取API密钥
        api_key = os.getenv('BAILIAN_API_KEY')
        
        if not api_key:
            logger.error("未找到BAILIAN_API_KEY环境变量")
            return None
            
        if api_key == 'sk':
            logger.warning("API密钥为默认值'sk'，请检查.env文件配置")
            return None
            
        if len(api_key) < 10:
            logger.warning("API密钥长度过短，可能配置有误")
            return None
            
        logger.info("API密钥加载成功")
        return api_key
        
    except FileNotFoundError:
        logger.error(".env文件不存在，请创建.env文件并配置BAILIAN_API_KEY")
        return None
    except Exception as e:
        logger.error(f"加载API密钥时发生错误: {str(e)}")
        return None

def validate_api_key(api_key):
    """
    验证API密钥格式
    
    Args:
        api_key (str): 要验证的API密钥
        
    Returns:
        bool: 验证是否通过
    """
    if not api_key:
        return False
        
    if api_key == 'sk':
        return False
        
    if len(api_key) < 10:
        return False
        
    # 可以添加更多验证规则
    if not api_key.startswith('sk-'):
        logger.warning("API密钥格式可能不正确，通常以'sk-'开头")
        
    return True

def get_api_key():
    """
    获取并验证API密钥
    
    Returns:
        str: 有效的API密钥，如果无效返回None
    """
    api_key = load_api_key()
    
    if validate_api_key(api_key):
        return api_key
    else:
        logger.error("API密钥验证失败")
        return None
