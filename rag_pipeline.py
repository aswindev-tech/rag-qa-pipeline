import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from transformers import pipeline

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline


load_dotenv()


class DocumentIngester:

    def load_text(self, file_path: str) -> List[Document]:
        return TextLoader(
            file_path,
            encoding="utf-8"
        ).load()

    def load_pdf(self, file_path: str) -> List[Document]:
        return PyPDFLoader(file_path).load()

    def load_directory(
        self,
        directory: str,
        pattern: str = "**/*.txt"
    ) -> List[Document]:

        loader = DirectoryLoader(
            directory,
            glob=pattern,
            show_progress=True,
            use_multithreading=True
        )

        return loader.load()

    def load_texts(
        self,
        texts: List[str],
        metadata: Optional[List[dict]] = None
    ) -> List[Document]:

        metadata = metadata or [{} for _ in texts]

        return [
            Document(
                page_content=text,
                metadata=meta
            )
            for text, meta in zip(texts, metadata)
        ]


class DocumentChunker:

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def split(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)


class VectorStore:

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu"
    ):

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )

        self.store = None

    def build(self, documents: List[Document]):

        self.store = FAISS.from_documents(
            documents,
            self.embeddings
        )

        return self.store

    def save(self, path: str):

        if self.store is None:
            raise RuntimeError("Vector store has not been created.")

        Path(path).mkdir(
            parents=True,
            exist_ok=True
        )

        self.store.save_local(path)

    def load(self, path: str):

        self.store = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        return self.store

    def retriever(self, top_k: int = 4):

        if self.store is None:
            raise RuntimeError("Vector store has not been created.")

        return self.store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k,
                "fetch_k": top_k * 3
            }
        )


def create_prompt():

    template = """
Answer the question using only the information provided in the context.

If the answer is not available in the context, say:
"I don't have enough information in the provided documents to answer that."

Keep the answer clear and concise.

Context:
{context}

Question:
{question}

Answer:
"""

    return PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )


class RAGPipeline:

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 4,
        device: str = "cpu",
        max_new_tokens: int = 256
    ):

        self.ingester = DocumentIngester()

        self.chunker = DocumentChunker(
            chunk_size,
            chunk_overlap
        )

        self.vector_store = VectorStore(
            embedding_model,
            device
        )

        self.top_k = top_k

        local_model = pipeline(
            "text2text-generation",
            model=model_name,
            max_new_tokens=max_new_tokens
        )

        self.llm = HuggingFacePipeline(
            pipeline=local_model
        )

        self.chain = None

    def ingest_texts(
        self,
        texts: List[str],
        metadata: Optional[List[dict]] = None
    ):

        documents = self.ingester.load_texts(
            texts,
            metadata
        )

        return self._build_index(documents)

    def ingest_file(self, file_path: str):

        extension = Path(file_path).suffix.lower()

        if extension == ".pdf":
            documents = self.ingester.load_pdf(file_path)

        elif extension == ".txt":
            documents = self.ingester.load_text(file_path)

        else:
            raise ValueError(
                "Only PDF and TXT files are supported."
            )

        return self._build_index(documents)

    def ingest_directory(
        self,
        directory: str,
        pattern: str = "**/*.txt"
    ):

        documents = self.ingester.load_directory(
            directory,
            pattern
        )

        return self._build_index(documents)

    def save_index(self, path: str):

        self.vector_store.save(path)

    def load_index(self, path: str):

        self.vector_store.load(path)

        self.chain = self._create_chain()

        return self

    def ask(self, question: str):

        if self.chain is None:
            raise RuntimeError(
                "No documents have been loaded."
            )

        result = self.chain.invoke({
            "query": question
        })

        return {
            "answer": result["result"].strip(),
            "sources": result.get(
                "source_documents",
                []
            )
        }

    def _build_index(self, documents):

        chunks = self.chunker.split(documents)

        self.vector_store.build(chunks)

        self.chain = self._create_chain()

        return self

    def _create_chain(self):

        retriever = self.vector_store.retriever(
            self.top_k
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": create_prompt()
            }
        )


SAMPLE_DOCUMENTS = [
    {
        "text": (
            "Artificial intelligence (AI) is the simulation of "
            "human intelligence in machines. AI systems can learn "
            "from data and perform tasks such as learning, "
            "reasoning and problem-solving."
        ),
        "source": "ai.txt"
    },
    {
        "text": (
            "Machine learning is a subset of artificial intelligence "
            "that allows computers to learn from data without being "
            "explicitly programmed. Common algorithms include "
            "linear regression, decision trees, random forests "
            "and support vector machines."
        ),
        "source": "machine_learning.txt"
    },
    {
        "text": (
            "Deep learning is a part of machine learning that uses "
            "neural networks with multiple layers. CNNs are commonly "
            "used for image processing, while RNNs and Transformers "
            "can be used for sequential and language tasks."
        ),
        "source": "deep_learning.txt"
    },
    {
        "text": (
            "Natural Language Processing (NLP) allows computers "
            "to understand and generate human language. Modern "
            "NLP systems commonly use Transformer architectures."
        ),
        "source": "nlp.txt"
    },
    {
        "text": (
            "Retrieval-Augmented Generation (RAG) combines "
            "information retrieval with language generation. "
            "Relevant information is retrieved from a knowledge "
            "base and provided to an LLM as context."
        ),
        "source": "rag.txt"
    },
    {
        "text": (
            "FAISS is a library for efficient similarity search "
            "of vector embeddings. It is commonly used in RAG "
            "systems to find documents that are semantically "
            "similar to a user's question."
        ),
        "source": "faiss.txt"
    }
]


def run_demo():

    print("\nRAG-Based Generative AI Application\n")

    rag = RAGPipeline(
        model_name="google/flan-t5-base",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=300,
        chunk_overlap=50,
        top_k=3
    )

    texts = [
        item["text"]
        for item in SAMPLE_DOCUMENTS
    ]

    metadata = [
        {"source": item["source"]}
        for item in SAMPLE_DOCUMENTS
    ]

    rag.ingest_texts(
        texts,
        metadata
    )

    questions = [
        "What is artificial intelligence?",
        "What is the difference between machine learning and deep learning?",
        "What is RAG?",
        "What is FAISS?",
        "What architecture is commonly used in modern NLP?"
    ]

    for question in questions:

        result = rag.ask(question)

        print(f"Question: {question}")
        print(f"Answer: {result['answer']}")

        if result["sources"]:
            print("Sources:")

            for document in result["sources"]:
                print(
                    f"- {document.metadata.get('source', 'unknown')}"
                )

        print("-" * 60)


if __name__ == "__main__":
    run_demo()