import hashlib
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os
from utils.logger_handler import  logger


def get_file_md5(file_path:str):
    """
    得到MD5文件
    :param file_path: 文件路径
    :return:
    """
    if not os.path.exists(file_path):
        logger.error(f"File {file_path} does not exist")
        return

    if not os.path.isfile(file_path):
        logger.error(f"File {file_path} is not a file")
        return

    md5_obj = hashlib.md5() # 创建MD5对象
    chunk_size = 4096 # 4kb防止文件过大
    try:
        with open(file_path, "rb") as f: # 二进制读取
            while chunk := f.read(chunk_size): # chunk为空停止循环
                md5_obj.update(chunk)
        md5_hex = md5_obj.hexdigest() # 转十六进制
        return md5_hex
    except Exception as e:
        logger.error(f"计算{file_path}失败，{str(e)}")
        return None

def listdir_with_allow_type(path:str, allow_type:tuple[str]):
    """
    判断文件类型是否为允许的类型
    :param path:
    :param allow_type:
    :return:
    """
    files = []
    if not os.path.isdir(path):
        logger.error(f"Path {path} is not a directory")
        return allow_type

    for f in os.listdir(path): # 列出该目录下所有文件名、文件夹名
        if f.endswith(allow_type): # 后缀
            files.append(os.path.join(path, f))
    return tuple(files)

def pdf_loader(file_path:str) -> list[Document]:
    """
    返回PDF文本
    :param file_path:
    :return:
    """
    return PyPDFLoader(file_path).load()

def text_loader (file_path:str) -> list[Document]:
    """
    返回text文本
    :param file_path:
    :return:
    """
    return TextLoader(file_path, encoding="utf-8").load()