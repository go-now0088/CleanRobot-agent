import os
import logging
from datetime import datetime

from utils.path_tool import get_abs_path

log_root = get_abs_path("logs")

os.makedirs(log_root, exist_ok=True) # 存在跳过，没有创建
default_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s') # 日志格式

def get_logger(
        name: str = "agent",
        console_level: int = logging.INFO, # 控制台级别
        file_level: int = logging.DEBUG, # 文件级别
        log_file: str = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers: # 避免重复创建
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(default_formatter)
    logger.addHandler(console_handler)

    if not log_file:
        log_file = os.path.join(log_root, f"{name}_{datetime.now().strftime('%Y-%m-%d')}.log") # 文件路径

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(default_formatter)
    logger.addHandler(file_handler)

    return logger

logger = get_logger()

if __name__ == '__main__':
    logger.info("信息日志")
    logger.warning("警告日志")
    logger.error("错误日志")
    logger.debug("调试日志")
