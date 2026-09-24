from contextlib import asynccontextmanager
import sqlite3

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import authenticate_user, create_tables, create_user


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    yield


app = FastAPI(title="LearnSmart API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    email: str
    password: str


class SignupRequest(BaseModel):
    name: str
    email: str
    password: str
    role: str


class UserResponse(BaseModel):
    name: str
    email: str
    role: str
    id: int


class AuthResponse(BaseModel):
    success: bool
    user: UserResponse


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/login", response_model=AuthResponse)
def login(request: LoginRequest):
    user = authenticate_user(request.email, request.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    return {"success": True, "user": user}


@app.post("/auth/signup", response_model=AuthResponse)
def signup(request: SignupRequest):
    try:
        user_id = create_user(
            request.name,
            request.email,
            request.password,
            request.role,
        )
    except sqlite3.IntegrityError as error:
        if "users.email" in str(error):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A user with this email already exists",
            ) from error
        raise

    user = authenticate_user(request.email, request.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User was created but could not be loaded",
        )

    user["id"] = user_id
    return {"success": True, "user": user}
