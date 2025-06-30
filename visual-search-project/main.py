# main.py - Complete Visual Search API with Authentication
# Properly ordered version

# ===== IMPORTS SECTION =====
from fastapi import FastAPI, HTTPException, File, UploadFile, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import chromadb
from transformers import CLIPProcessor, CLIPModel, pipeline
import torch
import numpy as np
from PIL import Image
import io
import os
import json
import logging
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import base64
from functools import lru_cache
import jwt
from passlib.context import CryptContext
from pathlib import Path

# ===== LOGGING CONFIGURATION =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== STARTUP AND SHUTDOWN EVENTS =====
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    logger.info("Starting Visual Search API...")
    
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    
    # Load models
    models_loaded = load_models()
    if not models_loaded:
        logger.error("Failed to load models!")
    
    # Load ChromaDB
    db_loaded = load_chromadb()
    if not db_loaded:
        logger.error("Failed to load ChromaDB!")
    
    if models_loaded and db_loaded:
        logger.info("✅ Visual Search API ready!")
    else:
        logger.error("❌ Failed to initialize some components")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Visual Search API...")

# Update the app creation to use lifespan
app = FastAPI(
    title="Visual Search API",
    description="Enterprise-grade visual search system with AI explanations and authentication",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add after the app creation
from fastapi import Request

# @app.middleware("http")
# async def log_requests(request: Request, call_next):
    # """Log all incoming requests for debugging"""
    # if request.url.path == "/search" and request.method == "POST":
        # body = await request.body()
        # logger.info(f"Raw request body: {body}")
        # # Recreate request with body
        # from starlette.requests import Request as StarletteRequest
        # request = StarletteRequest(request.scope, receive=lambda: {"type": "http.request", "body": body})
    
    # response = await call_next(request)
    # return response

# ===== GLOBAL VARIABLES =====
clip_model = None
clip_processor = None
chroma_collection = None
explanation_generator = None
device = None

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

# ===== CONFIGURATION CLASSES =====
# Get base directory for relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db_test")  # Update if needed
    COLLECTION_NAME = "image_embeddings_test_500"  # Update with your collection name
    MODEL_NAME = "openai/clip-vit-base-patch32"
    EXPLANATION_MODEL = "Salesforce/blip-image-captioning-base"
    MAX_RESULTS = 10
    DEFAULT_TOP_K = 5
    IMAGE_BASE_PATH = os.path.join(BASE_DIR, "images")  # Update if needed
    CACHE_SIZE = 100

class AuthConfig:
    SECRET_KEY = "your-secret-key-change-this-in-production"  # IMPORTANT: Change this!
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# ===== SECURITY SETUP =====
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Simple in-memory user storage (use a database in production)
users_db = {}

# ===== PYDANTIC MODELS =====
# Search models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(default=5, ge=1, le=Config.MAX_RESULTS, description="Number of results to return")
    include_explanations: bool = Field(default=True, description="Generate AI explanations for results")

class SearchResult(BaseModel):
    filename: str
    path: str
    similarity_score: float
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = {}

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    search_time_ms: float
    total_results: int

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    database_connected: bool
    total_images: int
    timestamp: str

# Authentication models
class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern="^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
    password: str = Field(..., min_length=6)

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str

class User(BaseModel):
    username: str
    email: str
    created_at: datetime

# ===== HELPER FUNCTIONS =====
# Authentication helpers
def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict) -> str:
    """Create a JWT token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, AuthConfig.SECRET_KEY, algorithm=AuthConfig.ALGORITHM)
    return encoded_jwt

def decode_token(token: str) -> dict:
    """Decode and verify a JWT token"""
    try:
        payload = jwt.decode(token, AuthConfig.SECRET_KEY, algorithms=[AuthConfig.ALGORITHM])
        return payload
    except jwt.JWTError:
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Get the current authenticated user"""
    token = credentials.credentials
    payload = decode_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    username = payload.get("sub")
    if username is None or username not in users_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return users_db[username]

# Model loading functions
@lru_cache(maxsize=1)
def load_models():
    """Load all required models with caching"""
    global clip_model, clip_processor, explanation_generator, device
    
    try:
        logger.info("Loading models...")
        
        # Detect device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load CLIP model
        logger.info(f"Loading CLIP model: {Config.MODEL_NAME}")
        clip_model = CLIPModel.from_pretrained(Config.MODEL_NAME).to(device)
        clip_processor = CLIPProcessor.from_pretrained(Config.MODEL_NAME)
        clip_model.eval()
        
        # Load explanation generator (BLIP model for image captioning)
        logger.info("Loading explanation generator...")
        explanation_generator = pipeline(
            "image-to-text", 
            model=Config.EXPLANATION_MODEL,
            device=0 if device == "cuda" else -1
        )
        
        logger.info("All models loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

def load_chromadb():
    """Load ChromaDB collection with fallback options"""
    global chroma_collection
    
    try:
        logger.info(f"Loading ChromaDB from {Config.CHROMA_DB_PATH}")
        
        # Try multiple approaches
        try:
            # Approach 1: Standard connection
            client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
            collections = client.list_collections()
            
            if collections:
                chroma_collection = collections[0]  # Use first available
                logger.info(f"Loaded collection: {chroma_collection.name}")
                return True
                
        except Exception as e:
            logger.warning(f"Standard connection failed: {e}")
            
            # Approach 2: Try direct reader
            try:
                from direct_parquet_reader import DirectEmbeddingReader
                chroma_collection = DirectEmbeddingReader(Config.CHROMA_DB_PATH)
                if chroma_collection.count() > 0:
                    logger.info(f"Loaded {chroma_collection.count()} embeddings using direct reader")
                    return True
            except:
                pass
            
            # Approach 3: Create mock collection for testing
            logger.warning("Creating mock collection for testing")
            
            class MockCollection:
                def count(self):
                    return 0
                
                def query(self, query_embeddings, n_results=5):
                    return {
                        'ids': [[]],
                        'distances': [[]],
                        'metadatas': [[]]
                    }
            
            chroma_collection = MockCollection()
            logger.warning("Using mock collection - searches will return no results")
            logger.warning("Please re-run your embedding generation script to fix this")
            return True
            
    except Exception as e:
        logger.error(f"All ChromaDB loading attempts failed: {e}")
        return False

# Search helper functions
def generate_text_embedding(text: str) -> List[float]:
    """Generate CLIP embedding for text query"""
    inputs = clip_processor(text=[text], return_tensors="pt").to(device)
    
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    
    return text_features.cpu().numpy().flatten().tolist()

# async def generate_explanation(image_path: str, query: str, similarity: float) -> str:
    # """Generate AI explanation for why image matches query"""
    # try:
        # # Load image
        # image = Image.open(image_path).convert("RGB")
        
        # # Generate caption using BLIP
        # caption = await asyncio.get_event_loop().run_in_executor(
            # executor,
            # lambda: explanation_generator(image)[0]['generated_text']
        # )
        
        # # Create contextual explanation
        # if similarity > 0.7:
            # confidence = "highly relevant"
        # elif similarity > 0.5:
            # confidence = "very relevant"
        # elif similarity > 0.3:
            # confidence = "relevant"
        # else:
            # confidence = "somewhat relevant"
        
        # explanation = f"This image is {confidence} to '{query}' because it shows {caption}. "
        
        # # Add specific details based on query keywords
        # query_lower = query.lower()
        # if any(word in query_lower for word in ['sunset', 'sunrise', 'sky']):
            # explanation += "The lighting and colors suggest atmospheric conditions mentioned in your search. "
        # elif any(word in query_lower for word in ['flower', 'plant', 'garden']):
            # explanation += "The botanical elements match your search criteria. "
        # elif any(word in query_lower for word in ['animal', 'bird', 'wildlife']):
            # explanation += "The fauna captured aligns with your query. "
        # elif any(word in query_lower for word in ['water', 'ocean', 'lake', 'river']):
            # explanation += "The aquatic features correspond to your search terms. "
        
        # explanation += f"(Similarity score: {similarity:.2%})"
        
        # return explanation
        
    # except Exception as e:
        # logger.error(f"Error generating explanation: {e}")
        # return f"This image matches your query '{query}' with {similarity:.2%} confidence."
        
async def generate_explanation(image_path: str, query: str, similarity: float) -> str:
    """Generate AI explanation for why image matches query with flexible path handling"""
    try:
        # Extract just the filename from the stored path
        filename = os.path.basename(image_path)
        
        # Construct the actual local path using current config
        actual_image_path = os.path.join(Config.IMAGE_BASE_PATH, filename)
        
        # Check if file exists at the new location
        if not os.path.exists(actual_image_path):
            logger.warning(f"Image not found: {actual_image_path}")
            # Return basic explanation without image analysis
            return f"This image matches your query '{query}' with {similarity:.2%} confidence based on stored embeddings."
        
        # Load image from the correct path
        image = Image.open(actual_image_path).convert("RGB")
        
        # Generate caption using BLIP
        caption = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: explanation_generator(image)[0]['generated_text']
        )
        
        # Create contextual explanation
        if similarity > 0.7:
            confidence = "highly relevant"
        elif similarity > 0.5:
            confidence = "very relevant"
        elif similarity > 0.3:
            confidence = "relevant"
        else:
            confidence = "somewhat relevant"
        
        explanation = f"This image is {confidence} to '{query}' because it shows {caption}. "
        
        # Add specific details based on query keywords
        query_lower = query.lower()
        if any(word in query_lower for word in ['sunset', 'sunrise', 'sky']):
            explanation += "The lighting and colors suggest atmospheric conditions mentioned in your search. "
        elif any(word in query_lower for word in ['flower', 'plant', 'garden']):
            explanation += "The botanical elements match your search criteria. "
        elif any(word in query_lower for word in ['animal', 'bird', 'wildlife']):
            explanation += "The fauna captured aligns with your query. "
        elif any(word in query_lower for word in ['water', 'ocean', 'lake', 'river']):
            explanation += "The aquatic features correspond to your search terms. "
        
        explanation += f"(Similarity score: {similarity:.2%})"
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating explanation for {image_path}: {e}")
        # Return fallback explanation
        filename = os.path.basename(image_path)
        return f"This image ({filename}) matches your query '{query}' with {similarity:.2%} confidence."

# Remove the old startup event decorator and function

# ===== API ROUTES =====
# Root and health check endpoints
@app.get("/")
async def serve_frontend():
    """Serve the main frontend application"""
    return FileResponse("static/index.html")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health and status"""
    try:
        total_images = chroma_collection.count() if chroma_collection else 0
        
        return HealthResponse(
            status="healthy",
            models_loaded=clip_model is not None,
            database_connected=chroma_collection is not None,
            total_images=total_images,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Authentication endpoints
@app.post("/auth/register", response_model=Token)
async def register(user_data: UserRegister):
    """Register a new user"""
    # Check if username already exists
    if user_data.username in users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    users_db[user_data.username] = {
        "username": user_data.username,
        "email": user_data.email,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow()
    }
    
    # Create access token
    access_token = create_access_token(data={"sub": user_data.username})
    
    return Token(
        access_token=access_token,
        username=user_data.username
    )

@app.post("/auth/login", response_model=Token)
async def login(user_data: UserLogin):
    """Login user"""
    # Verify user exists
    user = users_db.get(user_data.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Verify password
    if not verify_password(user_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Create access token
    access_token = create_access_token(data={"sub": user_data.username})
    
    return Token(
        access_token=access_token,
        username=user_data.username
    )

@app.get("/auth/me", response_model=User)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current user info"""
    return User(
        username=current_user["username"],
        email=current_user["email"],
        created_at=current_user["created_at"]
    )

# Add a test endpoint
@app.post("/test/search")
async def test_search(request_body: dict):
    """Test endpoint to debug search requests"""
    logger.info(f"Test search received: {request_body}")
    
    # Manually validate
    if 'query' not in request_body:
        return {"error": "Missing 'query' field", "received": request_body}
    
    # Try to create SearchRequest manually
    try:
        search_req = SearchRequest(**request_body)
        return {"success": True, "parsed": search_req.dict()}
    except Exception as e:
        return {"error": str(e), "received": request_body}
#@app.post("/search", response_model=SearchResponse)
# async def search_images(
    # request: SearchRequest,
    # current_user: Optional[dict] = None  # Make auth optional for now
# ):
    # """Search for images using natural language query"""
    # start_time = datetime.now()
    
    # # Debug logging
    # logger.info(f"Search request received: query='{request.query}', top_k={request.top_k}")
    
    # # Log search query if user is authenticated
    # if current_user:
        # logger.info(f"User {current_user['username']} searching for: '{request.query}'")
    
    # try:
        # # Validate models are loaded
        # if not clip_model or not chroma_collection:
            # raise HTTPException(status_code=503, detail="Models not loaded")
        
        # # Generate text embedding
        # logger.info(f"Processing search query: '{request.query}'")
        # query_embedding = await asyncio.get_event_loop().run_in_executor(
            # executor,
            # generate_text_embedding,
            # request.query
        # )
        
        # # Search in ChromaDB
        # results = chroma_collection.query(
            # query_embeddings=[query_embedding],
            # n_results=request.top_k
        # )
        
        # # Process results
        # search_results = []
        
        # if results['metadatas'][0]:
            # for idx, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
                # # Calculate similarity score (1 - distance for cosine)
                # similarity = 1 - distance
                
                # # Create result
                # result = SearchResult(
                    # filename=metadata['filename'],
                    # path=metadata['path'],
                    # similarity_score=float(similarity),
                    # metadata=metadata
                # )
                
                # # Generate explanation if requested
                # if request.include_explanations:
                    # result.explanation = await generate_explanation(
                        # metadata['path'],
                        # request.query,
                        # similarity
                    # )
                
                # search_results.append(result)
        
        # # Calculate search time
        # search_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # return SearchResponse(
            # query=request.query,
            # results=search_results,
            # search_time_ms=search_time,
            # total_results=len(search_results)
        # )
        
    # except Exception as e:
        # logger.error(f"Search error: {e}")
        # logger.error(f"Request data: {request}")
        # raise HTTPException(status_code=500, detail=str(e))
@app.post("/search", response_model=SearchResponse)
async def search_images(request: SearchRequest):
    """Search for images using natural language query"""
    start_time = datetime.now()
    
    # Debug logging
    logger.info(f"Search request received: query='{request.query}', top_k={request.top_k}")
    
    try:
        # Validate models are loaded
        if not clip_model or not chroma_collection:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Generate text embedding
        logger.info(f"Processing search query: '{request.query}'")
        query_embedding = await asyncio.get_event_loop().run_in_executor(
            executor,
            generate_text_embedding,
            request.query
        )
        
        # Search in ChromaDB
        results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=request.top_k
        )
        
        # Process results
        search_results = []
        
        if results['metadatas'][0]:
            for idx, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
                # Calculate similarity score (1 - distance for cosine)
                similarity = 1 - distance
                
                # Create result
                result = SearchResult(
                    filename=metadata['filename'],
                    path=metadata['path'],
                    similarity_score=float(similarity),
                    metadata=metadata
                )
                
                # Generate explanation if requested
                if request.include_explanations:
                    result.explanation = await generate_explanation(
                        metadata['path'],
                        request.query,
                        similarity
                    )
                
                search_results.append(result)
        
        # Calculate search time
        search_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            search_time_ms=search_time,
            total_results=len(search_results)
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        logger.error(f"Request data: {request}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/image/{filename}")
async def get_image(filename: str):
    """Retrieve a specific image by filename"""
    try:
        # Construct image path
        image_path = os.path.join(Config.IMAGE_BASE_PATH, filename)
        
        # Check if file exists
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"Image not found: {filename}")
        
        # Return image file
        return FileResponse(
            image_path,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"}
        )
        
    except Exception as e:
        logger.error(f"Error retrieving image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/image")
async def search_by_image(file: UploadFile = File(...), top_k: int = Query(default=5)):
    """Search using an uploaded image (reverse image search)"""
    try:
        # Read uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Generate image embedding
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        query_embedding = image_features.cpu().numpy().flatten().tolist()
        
        # Search similar images
        results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Process results
        search_results = []
        for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
            similarity = 1 - distance
            search_results.append({
                "filename": metadata['filename'],
                "similarity": float(similarity),
                "metadata": metadata
            })
        
        return {
            "message": "Reverse image search completed",
            "results": search_results,
            "total": len(search_results)
        }
        
    except Exception as e:
        logger.error(f"Reverse image search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/history")
async def get_search_history(
    current_user: dict = Depends(get_current_user),
    limit: int = Query(default=10, ge=1, le=50)
):
    """Get user's search history"""
    # In a real app, this would query a database
    # For now, return empty history
    return {
        "username": current_user["username"],
        "history": [],
        "message": "Search history feature coming soon"
    }

# ===== STATIC FILE SERVING =====
# Mount static files - this must be at the end
app.mount("/static", StaticFiles(directory="static"), name="static")

# ===== MAIN ENTRY POINT =====
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Visual Search API...")
    print("📁 Make sure to:")
    print("   1. Copy index.html to static/index.html")
    print("   2. Update Config paths (CHROMA_DB_PATH, IMAGE_BASE_PATH)")
    print("   3. Change AuthConfig.SECRET_KEY for production")
    print("\n📄 Access the app at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    
    # Run without reload for direct execution
    uvicorn.run(app, host="0.0.0.0", port=8000)