# RAG-Based AI Document Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF or TXT documents and ask questions about their content.

The application retrieves relevant information from uploaded documents using semantic search and generates grounded answers using a local language model.

---

## Project Overview

This project implements an end-to-end Retrieval-Augmented Generation pipeline for document-based question answering.

The system combines:

- Document processing
- Text chunking
- Semantic embeddings
- Vector similarity search
- Context retrieval
- Prompt engineering
- Local language model generation
- Source document retrieval

Users can upload a document through the Streamlit web interface, ask questions about its content, and receive an answer along with the relevant source documents.

---

## Key Features

- PDF and TXT document upload
- Automatic document text extraction
- Text chunking with overlapping chunks
- Semantic text embeddings
- FAISS vector similarity search
- Relevant context retrieval
- Retrieval-Augmented Generation
- Local FLAN-T5 language model
- Source document display
- Streamlit web interface
- Runs locally without paid LLM inference APIs

---

## System Architecture

```text
                         User
                           |
                           v
                  Streamlit Web App
                           |
                           v
                  Upload PDF / TXT
                           |
                           v
                   Document Loader
                           |
                           v
                    Text Chunking
                           |
                           v
              Sentence Transformer
                    Embeddings
                           |
                           v
                  FAISS Vector Store
                           |
                           v
                 Semantic Retrieval
                           |
                           v
                Relevant Document
                      Context
                           |
                           v
                     RAG Prompt
                           |
                           v
                   FLAN-T5 Base
                         LLM
                           |
                           v
                  Generated Answer
                           |
                           v
                   Answer + Sources
```

---

## RAG Pipeline

The application follows these steps:

### 1. Document Ingestion

PDF and TXT files are loaded and converted into document objects.

### 2. Text Chunking

Large documents are divided into smaller overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`.

### 3. Embedding Generation

Each document chunk is converted into a vector representation using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

### 4. Vector Storage

The generated embeddings are stored in a FAISS vector index.

### 5. Semantic Retrieval

When a user asks a question, the question is converted into an embedding and compared with the stored document embeddings.

The most relevant document chunks are retrieved.

### 6. Context Injection

The retrieved document chunks are added to the RAG prompt together with the user's question.

### 7. Answer Generation

The prompt is passed to:

```text
google/flan-t5-base
```

The model generates an answer using the retrieved document context.

### 8. Source Display

The application displays the source documents used during retrieval.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | RAG pipeline and orchestration |
| Hugging Face Transformers | Local language model |
| FLAN-T5 Base | Answer generation |
| Sentence Transformers | Text embeddings |
| FAISS | Vector similarity search |
| Streamlit | Web interface |
| PyPDF | PDF document processing |

---

## Project Structure

```text
rag-qa-pipeline/
│
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

### `app.py`

Streamlit web application that provides the interface for uploading documents and asking questions.

### `rag_pipeline.py`

Contains the complete RAG pipeline, including:

- Document ingestion
- Text chunking
- Embedding generation
- FAISS vector storage
- Semantic retrieval
- Prompt construction
- Local LLM generation
- Source retrieval

### `requirements.txt`

Contains the Python dependencies required to run the project.

### `.env.example`

Example environment configuration file.

### `.gitignore`

Prevents sensitive files, virtual environments, caches, and temporary files from being committed to GitHub.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/aswindev-tech/rag-qa-pipeline.git
```

### 2. Open the Project

```bash
cd rag-qa-pipeline
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

On Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## How to Use

1. Open the Streamlit application.
2. Upload a PDF or TXT document.
3. Click **Process Document**.
4. Enter a question related to the uploaded document.
5. Click **Ask AI**.
6. The system retrieves relevant document sections.
7. The local language model generates an answer.
8. The retrieved source documents are displayed below the answer.

---

## Example Questions

```text
What is artificial intelligence?
```

```text
What does the document say about machine learning?
```

```text
What are the main topics discussed in the document?
```

```text
What is the conclusion of the document?
```

---

## Local Language Model

The project uses:

```text
google/flan-t5-base
```

The model runs locally using Hugging Face Transformers.

No paid Hugging Face inference API is required for generating answers.

The first run downloads the model and stores it in the local Hugging Face cache.

---

## Embedding Model

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model converts document chunks and user questions into dense numerical vectors for semantic similarity search.

---

## Vector Database

FAISS is used for storing and searching document embeddings.

The system uses semantic retrieval to identify document chunks that are most relevant to the user's question.

---

## Security

Sensitive credentials are not included in the repository.

The following files and directories are excluded using `.gitignore`:

```text
.env
venv/
__pycache__/
*.pyc
```

Never commit API keys, tokens, passwords, or other credentials to GitHub.

---

## Current Limitations

- FLAN-T5 Base is a relatively small language model.
- Answer quality depends on the uploaded documents and retrieved context.
- The application currently supports PDF and TXT files.
- Large documents may require additional processing time and memory.
- The application currently runs locally.

---

## Future Improvements

- Support multiple documents
- Add conversation history
- Add a chat-style interface
- Improve document management
- Add persistent vector storage
- Add source content previews
- Improve retrieval evaluation
- Add answer quality evaluation
- Use a stronger local language model
- Add authentication
- Deploy the application online
- Add document deletion and management
- Improve the overall user interface

---

## Project Highlights

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation
- Large Language Models
- Natural Language Processing
- Semantic Search
- Vector Databases
- Document Question Answering
- Prompt Engineering
- Local AI Model Deployment
- LangChain
- Hugging Face
- FAISS
- Streamlit

---

## Author

**Aswin Dev**

GitHub:

https://github.com/aswindev-tech

---

## Repository

https://github.com/aswindev-tech/rag-qa-pipeline

---

## License

This project is intended for educational, portfolio, and demonstration purposes.
