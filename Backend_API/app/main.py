
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

from .database import engine
from app import models
from .routers import post, user,auth,vote
from fastapi import status
# models.Base.metadata.create_all(bind=engine)
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials= True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(post.router)
app.include_router(user.router)
app.include_router(auth.router)
app.include_router(vote.router)
        
@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    return {"messege": "Welcome to my api"}




