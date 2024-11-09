from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import shutil
from typing import Optional
import pdfplumber
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use your existing API key loading code - THIS IS THE PART WE'RE CHANGING
try:
    with open('../openaiapi/openai_api_key.txt', 'r') as file:
        openai_api_key = file.read().strip()
    os.environ["OPENAI_API_KEY"] = openai_api_key
    logger.info("Successfully loaded OpenAI API key")
except FileNotFoundError:
    logger.error("OpenAI API key file not found! Make sure it exists at ../openaiapi/openai_api_key.txt")
    raise Exception("OpenAI API key file not found at ../openaiapi/openai_api_key.txt")


# Initialize models
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini")

# Global variable to store the current vector store
current_vectorstore = None

def process_pdf(file_path: str) -> str:
    """Extract text from PDF."""
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise Exception(f"Failed to process PDF: {str(e)}")

def create_vectorstore(text: str):
    """Create vector store from text."""
    try:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        
        global current_vectorstore
        current_vectorstore = Chroma.from_texts(
            chunks,
            embeddings,
            collection_name="pdf_collection"
        )
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise Exception(f"Failed to create vector store: {str(e)}")

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    """Handle PDF upload."""
    logger.info(f"Receiving upload request for file: {file.filename}")
    
    if not file.filename.endswith('.pdf'):
        return JSONResponse(
            status_code=400,
            content={"detail": "Only PDF files are allowed"}
        )
    
    # Save uploaded file temporarily
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing PDF file: {file.filename}")
        text = process_pdf(temp_file_path)
        create_vectorstore(text)
        
        return JSONResponse(
            status_code=200,
            content={"message": "PDF processed successfully"}
        )
    
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error processing file: {str(e)}"}
        )
    
    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

class Question(BaseModel):
    text: str

@app.post("/ask/")
async def ask_question(question: Question):
    """Handle questions about the uploaded PDF."""
    logger.info(f"Received question: {question.text}")
    
    if current_vectorstore is None:
        return JSONResponse(
            status_code=400,
            content={"detail": "Please upload a PDF first"}
        )

    try:
        # Create retrieval chain
        retriever = current_vectorstore.as_retriever()
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_template("""
        You are a knowledgeable assistant analyzing a document. Use **only** the provided context to answer questions.
        When formatting your response:
        1. Use markdown formatting when appropriate (**, ##, etc.)
        2. Preserve any relevant headers, lists, or structured content
        3. Use bullet points or numbered lists when presenting multiple items
        4. Highlight key terms or concepts using **bold** when appropriate
                                                  
        Answer the following question based only on the provided context:
        Context: `{context}`
        Question: `{input}`

        Answer the question concisely and accurately, in a structured manner. If the answer cannot be found in the context, say "I cannot find the answer in the document."
        """)

        # Create and run the chain
        combine_documents_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, combine_documents_chain)

        response = retrieval_chain.invoke({"input": question.text})
        
        return JSONResponse(
            status_code=200,
            content={"answer": response["answer"]}
        )

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error processing question: {str(e)}"}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)