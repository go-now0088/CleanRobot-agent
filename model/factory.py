from abc import ABC, abstractmethod
from typing import Optional
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from utils.config_handler import  rag_config


class BaseModelFactory(ABC):
    """
    定义抽象类是子类必须实现
    """
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel] :
        pass

class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Embeddings | BaseChatModel:
        return DashScopeEmbeddings(model = rag_config["embedding_model_name"])

class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Embeddings | BaseChatModel:
        return ChatTongyi(model=rag_config["chat_model_name"])

chat_model = ChatModelFactory().generator()
embedding_model = EmbeddingsFactory().generator()