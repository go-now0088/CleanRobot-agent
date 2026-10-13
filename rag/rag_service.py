from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from model.factory import chat_model
from utils.prompt_loader import  load_rag_prompts
from langchain_core.prompts import PromptTemplate

def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt

class RagSummerizeService:
    def __init__(self):
        self.vectory_store = VectorStoreService()
        self.model = chat_model
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.retriever = self.vectory_store.get_retriever()
        self.chain = self.__init__chain()

    def __init__chain(self):
        chain = self.prompt_template | print_prompt  | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        """
        检索用户的内容
        :param query:
        :return:
        """
        return self.retriever.invoke(query)

    def rag_summerize_docs(self, query: str) -> str:
        """
        rag总结用户的内容在拼接知识库的内容，一起给模型
        :param query:
        :return:
        """
        context_docs = self.retriever_docs(query)
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"参考资料【{counter}】：{doc.page_content}, 参考元数据{doc.metadata}\n"
        return self.chain.invoke(
            {
                "input": query, # 用户问题
                "context": context # 拼接好的资料
            }
        )

if __name__ == '__main__':
    rag = RagSummerizeService()
    print(rag.rag_summerize_docs("别墅适合用什么机器人来打扫"))