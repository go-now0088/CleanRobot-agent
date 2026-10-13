from langchain_core.documents import Document
from utils.path_tool import get_abs_path
import os
from utils.config_handler import  chroma_config
from langchain_chroma import Chroma
from model.factory import embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.file_handler import  text_loader, pdf_loader, listdir_with_allow_type, get_file_md5
from utils.logger_handler import logger

class VectorStoreService:
    """
    向量存储服务
    """
    def __init__(self):
        self.vectory_store = Chroma(
            collection_name = chroma_config['collection_name'],
            embedding_function = embedding_model,
            persist_directory=chroma_config['persist_directory'],
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size = chroma_config['chunk_size'],
            chunk_overlap = chroma_config['chunk_overlap'],
            separators= chroma_config['separators'],
            length_function= len
        )

    def get_retriever(self):
        """
        检索器检索
        :return:
        """
        return self.vectory_store.as_retriever(search_kwargs = {"k": chroma_config['k']})

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return: None
        """

        def check_md5(md5_check:str):
            """
            检查MD5是否处理过
            :param md5_check:


            :return:
            """
            if not os.path.exists(get_abs_path(chroma_config['md5_hex_store'])): # 创建文件
                open(get_abs_path(chroma_config['md5_hex_store']), 'w', encoding="utf-8").close()
                return False
            with open(get_abs_path(chroma_config['md5_hex_store']), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_check:
                        return True
                return False

        def save_md5(md5_check:str):
            """
            保存MD5
            :param md5_check:
            :return:
            """
            with open(get_abs_path(chroma_config['md5_hex_store']), "a", encoding="utf-8") as f:
                f.write(md5_check + "\n")

        def get_file_document(read_path:str):
            """
            得到文件的内容
            :param read_path:
            :return:
            """
            if read_path.endswith("txt"):
                return text_loader(read_path)
            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            return []

        allowed_file_path = listdir_with_allow_type(
            get_abs_path(chroma_config["data_path"]),
            tuple(chroma_config["allow_knowledge_file_type"])
        )
        for path in allowed_file_path: # 获取文件的MD5
            md5_hex = get_file_md5(path)
            if check_md5(md5_hex):
                logger.info(f"{path}已经存在")
                continue
            try:
                documents: list[Document] = get_file_document(path)
                if not documents:
                    logger.warning(f"{path}内容为空")
                    continue

                spliter_documents: list[Document] = self.spliter.split_documents(documents)
                if not spliter_documents:
                    logger.warning(f"{path}分割后为空")
                    continue

                self.vectory_store.add_documents(spliter_documents)
                save_md5(md5_hex)
                logger.info(f"{path}加载成功")
            except Exception as e:
                logger.error(f"{path}加载失败, {str(e)}", exc_info=True)
                continue

if __name__ == '__main__':
    v = VectorStoreService()
    v.load_document()
    retriever = v.get_retriever()
    res = retriever.invoke("迷路")
    for doc in res:
        print(doc.page_content)