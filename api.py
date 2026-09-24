from contextlib import asynccontextmanager
import sqlite3

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import (
    authenticate_user,
    create_tables,
    create_user,
    get_student_profile,
    get_students_by_parent_email,
    get_students_by_school,
    get_students_by_teacher,
)


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


class TeacherStudentResponse(BaseModel):
    name: str
    class_name: str | None = None
    school_name: str | None = None
    parent_name: str | None = None
    overall_score: float | None = None
    attendance_percentage: float | None = None
    performance_level: str | None = None


class TeacherStudentsResponse(BaseModel):
    teacher_name: str
    students: list[TeacherStudentResponse]


class StudentProfileResponse(BaseModel):
    name: str
    class_name: str | None = None
    school_name: str | None = None
    teacher_name: str | None = None
    overall_score: float | None = None
    attendance_percentage: float | None = None
    study_hours_per_day: float | None = None
    assignment_score: float | None = None
    final_exam_score: float | None = None
    midterm_score: float | None = None
    participation_score: float | None = None


class ParentChildrenResponse(BaseModel):
    parent_email: str
    children: list[StudentProfileResponse]


class AdminStudentResponse(BaseModel):
    name: str
    class_name: str | None = None
    overall_score: float | None = None
    attendance_percentage: float | None = None
    performance_level: str | None = None
    teacher_name: str | None = None


class ClassSummaryResponse(BaseModel):
    class_name: str | None = None
    student_count: int
    average_score: float | None = None
    average_attendance: float | None = None


class AdminStudentsResponse(BaseModel):
    school_name: str
    students: list[AdminStudentResponse]
    classes: list[ClassSummaryResponse]


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


@app.get(
    "/teacher/{teacher_name}/students",
    response_model=TeacherStudentsResponse,
)
def teacher_students(teacher_name: str):
    records = get_students_by_teacher(teacher_name)
    return {
        "teacher_name": teacher_name,
        "students": [
            {
                "name": record["student_name"],
                "class_name": record["class_name"],
                "school_name": record["school_name"],
                "parent_name": (
                    None
                    if record["parent_name"] == "Not linked"
                    else record["parent_name"]
                ),
                "overall_score": record["overall_score"],
                "attendance_percentage": record["attendance_percentage"],
                "performance_level": record["performance_level"],
            }
            for record in records
        ],
    }


@app.get(
    "/student/{student_name}/profile",
    response_model=StudentProfileResponse,
)
def student_profile(student_name: str):
    record = get_student_profile(student_name)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student '{student_name}' was not found",
        )

    return {
        "name": record["student_name"],
        "class_name": record["class_name"],
        "school_name": record["school_name"],
        "teacher_name": record["teacher_name"],
        "overall_score": record["overall_score"],
        "attendance_percentage": record["attendance_percentage"],
        "study_hours_per_day": record["study_hours_per_day"],
        "assignment_score": record["assignment_score"],
        "final_exam_score": record["final_exam_score"],
        "midterm_score": record["midterm_score"],
        "participation_score": record["participation_score"],
    }


@app.get(
    "/parent/{parent_email}/children",
    response_model=ParentChildrenResponse,
)
def parent_children(parent_email: str):
    records = get_students_by_parent_email(parent_email)
    return {
        "parent_email": parent_email,
        "children": [
            {
                "name": record["student_name"],
                "class_name": record["class_name"],
                "school_name": record["school_name"],
                "teacher_name": record["teacher_name"],
                "overall_score": record["overall_score"],
                "attendance_percentage": record["attendance_percentage"],
                "study_hours_per_day": record["study_hours_per_day"],
                "assignment_score": record["assignment_score"],
                "final_exam_score": record["final_exam_score"],
                "midterm_score": record["midterm_score"],
                "participation_score": record["participation_score"],
            }
            for record in records
        ],
    }


@app.get(
    "/admin/{school_name}/students",
    response_model=AdminStudentsResponse,
)
def admin_students(school_name: str):
    records = get_students_by_school(school_name)
    class_records = {}

    for record in records:
        class_name = record["class_name"]
        summary = class_records.setdefault(
            class_name,
            {
                "class_name": class_name,
                "student_count": 0,
                "scores": [],
                "attendance": [],
            },
        )
        summary["student_count"] += 1
        if record["overall_score"] is not None:
            summary["scores"].append(record["overall_score"])
        if record["attendance_percentage"] is not None:
            summary["attendance"].append(record["attendance_percentage"])

    classes = [
        {
            "class_name": summary["class_name"],
            "student_count": summary["student_count"],
            "average_score": (
                sum(summary["scores"]) / len(summary["scores"])
                if summary["scores"]
                else None
            ),
            "average_attendance": (
                sum(summary["attendance"]) / len(summary["attendance"])
                if summary["attendance"]
                else None
            ),
        }
        for summary in class_records.values()
    ]

    return {
        "school_name": school_name,
        "students": [
            {
                "name": record["student_name"],
                "class_name": record["class_name"],
                "overall_score": record["overall_score"],
                "attendance_percentage": record["attendance_percentage"],
                "performance_level": record["performance_level"],
                "teacher_name": record["teacher_name"],
            }
            for record in records
        ],
        "classes": classes,
    }
