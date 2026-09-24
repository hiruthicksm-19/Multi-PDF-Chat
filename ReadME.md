# 📚 Multi-PDF Chat with Gemini

An AI-powered PDF question-answering application built with **Streamlit, Google Gemini, LangChain, and FAISS**.

The application allows users to upload multiple PDF documents, process their contents, and ask questions about the uploaded documents. It uses **Gemini embeddings + FAISS similarity search** to retrieve relevant content and **Gemini** to generate answers based on the retrieved context.

---

## 🚀 Features

- 📄 Upload multiple PDF files
- 🔍 Extract text from PDF documents
- ✂️ Split large documents into smaller text chunks
- 🧠 Generate embeddings using Google's Gemini Embedding model
- 🔎 Perform semantic similarity search using FAISS
- 🤖 Generate answers using Google Gemini
- 💬 Ask questions directly about uploaded PDFs
- 🖥️ Simple Streamlit web interface
- 🔐 API key stored securely using environment variables

---

## 🏗️ Architecture

```text
                ┌─────────────────┐
                │   PDF Upload    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   PyPDF2        │
                │ Text Extraction │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Text Splitter   │
                │ LangChain       │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Gemini          │
                │ Embeddings      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │      FAISS      │
                │ Vector Database │
                └─────────────────┘


User Question
      │
      ▼
┌─────────────────┐
│ Gemini Embedding│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ FAISS Similarity│
│     Search      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Relevant PDF    │
│    Context      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Gemini LLM     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Generated Answer│
└─────────────────┘