import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE-API-KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file")
    st.stop()


def get_pdf_text(pdf_docs):

    text = ""

    for pdf in pdf_docs:

        pdf_reader = PdfReader(pdf)

        print("Pages:", len(pdf_reader.pages))

        for page in pdf_reader.pages:

            extracted_text = page.extract_text()

            if extracted_text:
                text += extracted_text + "\n"

    print("Extracted characters:", len(text))

    return text


def get_text_chunks(text):

    print("Text length:", len(text))

    if not text.strip():
        st.error(
            "No text could be extracted from the PDF. "
            "The PDF may be scanned/image-based."
        )
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    print("Number of chunks:", len(chunks))

    return chunks


def get_vectorstore(text_chunks):

    print("Number of chunks:", len(text_chunks))

    if not text_chunks:
        st.error("No text chunks were created from the PDF.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    vectorstore.save_local("faiss_index")

    print("FAISS index created successfully")


def get_conversation_chain():

    prompt_template = """
Answer the question as detailed as possible from the provided context.

If the answer is not available in the provided context, say:
"Answer is not available in the context."

Do not provide information that is not present in the context.

Context:
{context}

Question:
{question}

Answer:
"""

    model = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    return model, prompt


def user_input(user_question):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = new_db.similarity_search(user_question)

    model, prompt = get_conversation_chain()

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    final_prompt = prompt.format(
        context=context,
        question=user_question
    )

    response = model.invoke(final_prompt)

    if isinstance(response.content, str):
        answer = response.content
    else:
        answer = "".join(
            item.get("text", "")
            for item in response.content
            if isinstance(item, dict)
            and item.get("type") == "text"
        )

    st.write(answer)



def main():

    st.set_page_config(
        page_title="Chat With Multiple PDF",
        page_icon="📚"
    )

    st.header("Chat with Multiple PDF using Gemini 🤖")

    user_question = st.text_input(
        "Ask a Question from the PDF Files"
    )

    if user_question:
        if os.path.exists("faiss_index"):
            user_input(user_question)
        else:
            st.warning(
                "Please upload and process a PDF before asking questions."
            )

    with st.sidebar:

        st.title("Menu:")

        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Submit & Process"):

            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
                return

            with st.spinner("Processing..."):

                raw_text = get_pdf_text(pdf_docs)

                text_chunks = get_text_chunks(raw_text)

                get_vectorstore(text_chunks)

                st.success("PDFs processed successfully!")


if __name__ == "__main__":
    main()