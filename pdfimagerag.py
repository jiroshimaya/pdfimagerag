import argparse
import base64

import dotenv
import fitz  # pymupdf
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ParentDocumentRetriever
from langchain.schema import BaseRetriever, Document
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()


class PDFManager:
    def __init__(self, pdf_path):
        self.doc = fitz.open(pdf_path)
        self.text_by_page = self.extract_text_from_pdf()
        self.base64_images_by_page = self.convert_pdf_to_base64_images()
        self.retriever = self.create_retriever()

    def extract_text_from_pdf(self) -> list[str]:
        text_by_page = []
        for page_num in range(len(self.doc)):
            page = self.doc.load_page(page_num)
            text = page.get_text()
            text_by_page.append(text)
        return text_by_page

    def create_retriever(self) -> BaseRetriever:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = Chroma(embedding_function=embeddings)
        docstore = InMemoryStore()
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000, chunk_overlap=0
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250, chunk_overlap=200
        )
        retriever = ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=docstore,
            parent_splitter=parent_splitter,
            child_splitter=child_splitter,
        )

        documents = [
            Document(page_content=text, metadata={"page": str(page_num)})
            for page_num, text in enumerate(self.text_by_page)
        ]
        retriever.add_documents(documents)

        return retriever

    def convert_pdf_to_base64_images(self) -> list[str]:
        base64_images_by_page = []
        for page_num in range(len(self.doc)):
            page = self.doc.load_page(page_num)
            pix = page.get_pixmap()
            base64_image = base64.b64encode(pix.tobytes()).decode("utf-8")
            base64_images_by_page.append(base64_image)
        return base64_images_by_page

    def doc2base64(self, documents: list[Document]) -> list[str]:
        pages = [int(document.metadata["page"]) for document in documents]
        base64_images = [self.base64_images_by_page[page] for page in pages]
        return base64_images

    @property
    def rag_prompt_template(self) -> ChatPromptTemplate:
        messages = [
            ("system", "参考情報に基づいて質問に回答してください."),
            (
                "user",
                [
                    {"type": "text", "text": "{question}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            ),
        ]
        return ChatPromptTemplate.from_messages(messages)

    def ask(self, question: str) -> str:
        def search(question: str) -> str:
            docs = self.retriever.get_relevant_documents(question)
            base64_images = self.doc2base64(docs)
            return base64_images[0]

        chain = (
            {
                "question": RunnablePassthrough(),
                "base64_image": RunnableLambda(search),
            }
            | self.rag_prompt_template
            | ChatOpenAI(model="gpt-4o-mini")
        )
        return chain.invoke(question).content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDF to images using pymupdf")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("question", help="Question to ask")
    args = parser.parse_args()
    pdf_manager = PDFManager(args.pdf_path)
    print(pdf_manager.ask(args.question))
