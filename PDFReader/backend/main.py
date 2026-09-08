import os
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PyPDF2 import PdfReader

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv



# APP

app = FastAPI(
    title="PDF Question Answer API",
    description="Upload a PDF and ask questions about it",
    version="1.0.0"
)


# CORS


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



load_dotenv()

#  API KEY

if not os.getenv("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY is not set.")


# GLOBAL VECTOR DATABASE


vector_store = None
document_name = None

# MODELS


class QuestionRequest(BaseModel):
    question: str


# HEALTH CHECK


@app.get("/")
def home():
    return {
        "message": "PDF Question Answer API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }



# UPLOAD PDF

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    global vector_store
    global document_name

    # Check file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    try:

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            shutil.copyfileobj(
                file.file,
                temp_file
            )

            temp_file_path = temp_file.name

        # Read PDF
        pdf_reader = PdfReader(temp_file_path)

        all_text = ""

        for page_number, page in enumerate(pdf_reader.pages):

            text = page.extract_text()

            if text:
                all_text += text + "\n"

        # Delete temporary file
        os.remove(temp_file_path)

        # Check extracted text
        if not all_text.strip():

            raise HTTPException(
                status_code=400,
                detail="Could not extract text from this PDF."
            )

        
        # TEXT SPLITTING
        

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = text_splitter.split_text(
            all_text
        )

        
        # EMBEDDINGS
        

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        
        # FAISS
        

        vector_store = FAISS.from_texts(
            chunks,
            embeddings
        )

        document_name = file.filename

        return {
            "success": True,
            "filename": file.filename,
            "pages": len(pdf_reader.pages),
            "chunks": len(chunks),
            "message": "PDF uploaded and processed successfully."
        }

    except HTTPException:
        raise

    except Exception as e:

        print("Upload error:", e)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ASK QUESTION


@app.post("/ask")
async def ask_question(request: QuestionRequest):

    global vector_store
    global document_name

    # Check PDF uploaded
    if vector_store is None:

        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF first."
        )

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        # SEARCH RELEVANT DOCUMENT CHUNKS

        documents = vector_store.similarity_search(
            question,
            k=4
        )

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        # OPENAI CHAT MODEL

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

        prompt = f"""
You are a helpful PDF assistant.

Answer the user's question using ONLY the information
contained in the PDF context below.

If the answer is not available in the PDF, respond with ai response:



Do not invent information.

PDF CONTEXT:
----------------
{context}
----------------

USER QUESTION:
{question}

ANSWER:
"""

        response = llm.invoke(prompt)

        return {
            "success": True,
            "question": question,
            "answer": response.content,
            "document": document_name
        }

    except Exception as e:

        print("Question error:", e)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )