from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import asyncio
import os
import shutil
import logging
from dotenv import load_dotenv
import time
import uuid
from typing import List, Dict, Any

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferWindowMemory

# Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec

# PDF handling
import fitz  # PyMuPDF for PDF handling

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
    allow_origins=["http://localhost:3000", "https://spaces-ai-assistant.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini")

# Queue for document processing
document_queue = []
processing_status = {}

# Session-based conversation memory
session_memories = {}

def process_pdf(file_path: str, document_id: str, session_id: str) -> list:
    """Extract text from PDF and return as Document objects with session tracking."""
    docs = []
    try:
        with fitz.open(file_path) as pdf:
            for i, page in enumerate(pdf):
                text = page.get_text()
                if text:
                    doc_obj = Document(
                        page_content=text,
                        metadata={
                            "page": i + 1,
                            "source": file_path,
                            "document_id": document_id,
                            "session_id": session_id  # Add session tracking
                        }
                    )
                    docs.append(doc_obj)
        return docs
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise Exception(f"Failed to process PDF: {str(e)}")

def create_vectorstore(docs: list):
    """Create vector store from documents."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(docs)
        
        vectorstore = PineconeVectorStore(
            embedding=embeddings,
            index_name=INDEX_NAME,
            text_key="text"
        )

        vectorstore.add_texts(
            texts=[t.page_content for t in texts],
            metadatas=[t.metadata for t in texts],
            ids=[f"doc_{t.metadata['document_id']}_{i}" for i, t in enumerate(texts)]
        )
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise Exception(f"Failed to create vector store: {str(e)}")

async def process_document(temp_file_path: str, document_id: str, task_id: str, session_id: str):
    """Process document in background."""
    try:
        processing_status[task_id] = "processing"
        docs = process_pdf(temp_file_path, document_id, session_id)
        create_vectorstore(docs)
        processing_status[task_id] = "completed"
    except Exception as e:
        processing_status[task_id] = "failed"
        logger.error(f"Error processing document {document_id}: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/upload/")
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """Handle PDF upload with efficient queue processing."""
    logger.info(f"Receiving upload request for file: {file.filename} from session: {session_id}")
    
    if not file.filename.endswith('.pdf'):
        return JSONResponse(
            status_code=400,
            content={"detail": "Only PDF files are allowed"}
        )
    
    document_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    temp_file_path = f"temp_{document_id}_{file.filename}"
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        document_queue.append((temp_file_path, document_id, task_id, session_id))
        processing_status[task_id] = "queued"
        
        # Start processing in background
        asyncio.create_task(process_document(temp_file_path, document_id, task_id, session_id))
        
        return JSONResponse(
            status_code=202,
            content={
                "message": "PDF queued for processing",
                "task_id": task_id
            }
        )
    except Exception as e:
        logger.error(f"Error queuing upload: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error queuing file: {str(e)}"}
        )

@app.get("/process_status/{task_id}")
async def get_processing_status(task_id: str):
    """Check the processing status of a document."""
    status = processing_status.get(task_id, "not_found")
    return {"status": status}

class Question(BaseModel):
    text: str
    session_id: str

@app.post("/ask/")
async def ask_question(question: Question):
    """Handle questions about the uploaded PDFs."""
    logger.info(f"Received question: {question.text} from session: {question.session_id}")
    
    try:
        memory = session_memories.get(question.session_id)
        if not memory:
            memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",  # Add this to specify which key to store
                window_size=20
            )
            session_memories[question.session_id] = memory

        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text"
        )
        
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 200,
                "filter": {"session_id": question.session_id}
            }
        )

        prompt = ChatPromptTemplate.from_template("""
        **System Instructions:**
        You are "Spaces" - an AI Research Assistant designed to help users answer questions and engage in conversations strictly based on the provided knowledge base and conversation history. Your responses should adhere to the following guidelines:

        1. **Data Sources:**
        - **Context:** You will receive relevant information from a knowledge base, provided separately as `**context**.
        - **History:** You will also have access to the past conversation history, provided separately as `**conversation history** 
        2. **Response Guidelines:**
        - **Primary Sources:** Base all answers solely on the information contained within `{context}` and `{history}`.
        - **Exclusions:** Do not use any external knowledge or information beyond what is provided in **context** and **conversation history** .
        - **Acknowledgment of Limitations:** If the information required to answer a question is not available in **context** and **conversation history**, respond by stating that the information is not available.
        - **Handling General Queries:**
            - For general greetings or questions about your identity, provide standard responses.
            - **Examples:**
            - **Greeting:** "Hello! How can I assist you today?"
            - **Identity Inquiry:** "I am your AI Research Assistant, here to help you with information from your provided knowledge base."
        3. **Style and Tone:**
        - **Format:** All responses must be in **Markdown**.
        - **Clarity:** Provide clear and concise answers.
        - **Professionalism:** Maintain a professional and courteous tone.
        - **Objectivity:** Present information factually without personal opinions or biases.
        4. **Avoiding Hallucinations:**
        - Do not fabricate information. Ensure all responses can be directly linked to **context** OR **conversation history** 
        5. **Expressing Uncertainty:**
        - If a response requires information not present in **context**` or **history**, clearly state the lack of sufficient data.
        - **Example:** "I'm sorry, but I don't have the information on that topic in the provided materials."
        6. **Security and Privacy:**
        - Do not attempt to infer or reveal any sensitive information beyond the provided **context** and **conversation history**. 

        7. **Formatting:**
        - Use Markdown formatting to enhance readability.
        - Utilize bullet points, numbered lists, headings, and other Markdown elements as appropriate.

        ###User Interaction Instructions:###
        When a user poses a question or engages in conversation:
        1. **Refer to the Relevant Context:**
        - Utilize the **context** and **conversation history** to find pertinent information related to the user's query.
        2. **Generate Response:**
        - Formulate a response based solely on the retrieved information.
        - Ensure the response aligns with the guidelines outlined above.
        3. **Fallback for Unrelated Queries:**
        - For questions or statements outside the scope of the provided data (excluding greetings or identity questions), inform the user that the information is not available.
        - **Example:** "I can help you with information from your provided knowledge base. Do you have any specific questions related to that?"

        ###Example Scenarios:###
        1. **User Question Within Context:**
        - **User:** "Can you explain the main findings of the latest research on renewable energy from our database?"
        - **Assistant:** 
            ```markdown
            The main findings of the latest research on renewable energy are:
            1. **Increased Efficiency:** Advances in solar panel technology have led to a 20% increase in energy conversion efficiency.
            2. **Cost Reduction:** The cost of wind energy has decreased by 15% over the past year due to improved turbine designs.
            3. **Storage Solutions:** New battery storage systems have extended the viability of renewable energy sources by enabling longer storage periods.
            ```

        2. **User Asks for Unsupported Information:**
        - **User:** "What are your thoughts on the current political climate?"
        - **Assistant:** 
            ```markdown
            I'm here to assist you with information from your provided knowledge base. Let me know if you have any questions related to that.
            ```

        3. **User Greets the Assistant:**
        - **User:** "Hello!"
        - **Assistant:** 
            ```markdown
            Hello! How can I assist you today?
            ```**context** and **conversation history** 

        ###Data for Context###
        **Previous conversation history:**
        `{chat_history}`

        **User's Question:** `{question}`

        **Context from Knowledge Base:** {context}

        **Your Response:**
        """)

        retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
        )

        # Update how we pass the input
        response = retrieval_chain({
            "question": question.text,
        })

        answer = response['answer']

        return JSONResponse(
            status_code=200,
            content={
                "answer": answer,
                "sources": list(set(doc.metadata['document_id'] for doc in response.get('source_documents', [])))
            }
        )   

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error processing question: {str(e)}"}
        )

@app.post("/clear_conversation/")
async def clear_conversation(session_id: str):
    if session_id in session_memories:
        del session_memories[session_id]
        return {"message": "Conversation history cleared"}
    return {"message": "No conversation history found for this session"}

@app.post("/cleanup_old_sessions/")
async def cleanup_old_sessions(max_age_hours: int = 24):
    try:
        current_time = time.time()
        old_sessions = []
        
        for session_id, memory in session_memories.items():
            if hasattr(memory, 'last_access_time'):
                if (current_time - memory.last_access_time) > (max_age_hours * 3600):
                    old_sessions.append(session_id)
        
        for session_id in old_sessions:
            if session_id in session_memories:
                del session_memories[session_id]
        
        return {"message": f"Cleaned up {len(old_sessions)} old sessions"}
    except Exception as e:
        logger.error(f"Error during session cleanup: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error cleaning up sessions: {str(e)}"}
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))