from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import shutil
import fitz  # PyMuPDF for PDF handling
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import logging
from dotenv import load_dotenv
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Get API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

if not all([openai_api_key, pinecone_api_key, pinecone_environment]):
    logger.error("Missing required environment variables!")
    raise Exception("Please set all required environment variables (OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT)")

# Initialize OpenAI
os.environ["OPENAI_API_KEY"] = openai_api_key
logger.info("Successfully loaded OpenAI API key")

try:
    # Initialize Pinecone
    pc = PineconeClient(api_key=pinecone_api_key)
    INDEX_NAME = "pdf-qa-index"
    index = pc.Index(INDEX_NAME)

except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {str(e)}")
    raise Exception(f"Pinecone initialization failed: {str(e)}")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend-url.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini")

def process_pdf(file_path: str) -> list:
    """Extract text from PDF and return as Document objects."""
    docs = []
    try:
        with fitz.open(file_path) as pdf:
            for i, page in enumerate(pdf):
                text = page.get_text()
                if text:
                    doc_obj = Document(
                        page_content=text,
                        metadata={"page": i + 1, "source": file_path}
                    )
                    docs.append(doc_obj)
        return docs
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise Exception(f"Failed to process PDF: {str(e)}")
    
def clear_or_create_pinecone_index():
    """Clears all vectors from the specified Pinecone index or creates it if it does not exist."""
    try:
        if INDEX_NAME not in pc.list_indexes():
            # Create the index if it does not exist
            pc.create_index(
                name=INDEX_NAME,
                dimension=1536,  # Adjust to match your embedding model’s dimensionality
                metric='cosine'
            )
            logger.info(f"Created Pinecone index: {INDEX_NAME}")
        else:
            # Clear existing vectors in the index
            index = pc.Index(INDEX_NAME)
            index.delete(delete_all=True)
            logger.info(f"Successfully cleared all vectors from index: {INDEX_NAME}")
    except Exception as e:
        logger.error(f"Failed to clear or create Pinecone index {INDEX_NAME}: {str(e)}")
        raise Exception(f"Pinecone index management failed: {str(e)}")
    
def create_vectorstore(docs: list):
    """Create vector store from documents using Pinecone with optimized metadata."""
    try:
        # Ensure the index is ready (clear if exists or create if not)
        clear_or_create_pinecone_index()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(docs)

        vectorstore = PineconeVectorStore(
            embedding=embeddings,
            index_name=INDEX_NAME,
            text_key="text"
        )

        vectorstore.add_texts(
            texts=[t.page_content for t in texts],
            ids=[f"doc_{i}" for i in range(len(texts))]
        )
        logger.info("Successfully created vector store with new document vectors.")
        return vectorstore
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
    
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        docs = process_pdf(temp_file_path)
        create_vectorstore(docs)
        
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
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

class Question(BaseModel):
    text: str

@app.post("/ask/")
async def ask_question(question: Question):
    """Handle questions about the uploaded PDF."""
    logger.info(f"Received question: {question.text}")
    
    try:
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text"
        )
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 200})  # Adjusted k value

        # Combined system and human message in prompt template
        prompt = ChatPromptTemplate.from_template("""
        You are a knowledgeable assistant analyzing a document. Use **only** the provided context to answer questions.
        When formatting your response:
        1. Use markdown formatting when appropriate (**, ##, etc.)
        2. Preserve any relevant headers, lists, or structured content
        3. Use bullet points or numbered lists when presenting multiple items
        4. Highlight key terms or concepts using **bold** when appropriate

        Answer concisely, preserving structure and highlighting key points. If you cannot find the answer in the context, simply say - "I could not find the answer in the document."

        Context: ```{context}```
        Question: ```{input}```
        """)

        # Create and run the chain
        combine_documents_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, combine_documents_chain)

        # Retry mechanism
        max_retries = 2
        retry_delay = 2  # seconds
        for attempt in range(max_retries):
            response = retrieval_chain.invoke({"input": question.text})
            if response["answer"] != "I could not find the answer in the document.":
                return JSONResponse(
                    status_code=200,
                    content={"answer": response["answer"]}
                )
            logger.warning("Answer not found on attempt {attempt+1}, retrying...")
            time.sleep(retry_delay)
        
        # Final response if retries fail
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
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))