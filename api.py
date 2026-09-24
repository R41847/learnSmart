from contextlib import asynccontextmanager
import asyncio
import os
from pathlib import Path
import sqlite3
import sys

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import (
    authenticate_user,
    create_tables,
    create_user,
    create_assignment,
    get_assignment,
    get_assignment_questions,
    get_assignment_submission,
    get_student_profile,
    list_assignments_by_teacher,
    list_assignments_for_student,
    save_assignment_submission,
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


class LearningPlanRequest(BaseModel):
    student_name: str
    age: int
    preferred_topic: str
    learning_style: str


class LearningPlanResponse(BaseModel):
    success: bool
    plan: str


class AssignmentQuestionRequest(BaseModel):
    question: str
    model_answer: str
    max_points: float = 1


class CreateAssignmentRequest(BaseModel):
    teacher_name: str
    title: str
    subject: str
    instructions: str
    questions: list[AssignmentQuestionRequest]


class AssignmentQuestionResponse(BaseModel):
    id: int
    question_order: int
    question: str
    max_points: float


class AssignmentResponse(BaseModel):
    id: int
    title: str
    subject: str | None = None
    created_at: str


class StudentAssignmentResponse(AssignmentResponse):
    status: str


class AssignmentSubmissionRequest(BaseModel):
    student_name: str
    answers: list[str]


class AssignmentSubmissionResponse(BaseModel):
    answers: list[str]
    grading: dict
    earned_points: float
    max_points: float
    percentage: float
    submitted_at: str


def _generate_learning_plan(student_profile):
    """Load the RAG pipeline lazily and run its synchronous Gemini call."""
    rag_path = str(Path(__file__).resolve().parent / "Rag")
    if rag_path not in sys.path:
        sys.path.insert(0, rag_path)

    from genai import generate_learning_plan

    return generate_learning_plan(student_profile)


def _grade_assignment(title, questions, answers):
    rag_path = str(Path(__file__).resolve().parent / "Rag")
    if rag_path not in sys.path:
        sys.path.insert(0, rag_path)

    from genai import grade_assignment

    return grade_assignment(title, questions, answers)


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


@app.post(
    "/ai-assistant/generate-plan",
    response_model=LearningPlanResponse,
)
async def generate_plan(request: LearningPlanRequest):
    if request.age <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Age must be greater than zero",
        )

    record = get_student_profile(request.student_name)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student '{request.student_name}' was not found",
        )

    score_fields = {
        "Assignment": record["assignment_score"],
        "Midterm": record["midterm_score"],
        "Final Exam": record["final_exam_score"],
        "Participation": record["participation_score"],
    }
    available_scores = {
        name: score for name, score in score_fields.items()
        if score is not None
    }
    weakest_area = (
        min(available_scores, key=available_scores.get)
        if available_scores
        else "overall"
    )

    student_profile = {
        "performance_level": record["performance_level"] or "Unknown",
        "age": request.age,
        "education_level": "Primary",
        "learning_style": request.learning_style,
        "preferred_topics": request.preferred_topic,
        "weak_areas": f"Low {weakest_area} performance",
        "study_hours": record["study_hours_per_day"],
        "attendance": record["attendance_percentage"],
    }

    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is unavailable because GEMINI_API_KEY is not set",
        )

    try:
        plan = await asyncio.to_thread(
            _generate_learning_plan,
            student_profile,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AI service configuration error: {error}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI plan generation failed: {error}",
        ) from error

    if not isinstance(plan, str) or not plan.strip():
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI service returned an empty learning plan",
        )

    return {"success": True, "plan": plan}


@app.post("/assignments/create")
def assignment_create(request: CreateAssignmentRequest):
    if not request.questions:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="An assignment must include at least one question",
        )
    if any(question.max_points <= 0 for question in request.questions):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Each question max_points must be greater than zero",
        )

    assignment_id = create_assignment(
        request.teacher_name,
        request.title,
        request.subject,
        request.instructions,
        [question.model_dump() for question in request.questions],
    )
    return {
        "success": True,
        "assignment": {
            "id": assignment_id,
            "title": request.title,
            "subject": request.subject,
        },
    }


@app.get(
    "/assignments/teacher/{teacher_name}",
    response_model=list[AssignmentResponse],
)
def teacher_assignments(teacher_name: str):
    return [
        {
            "id": row[0],
            "title": row[1],
            "subject": row[2],
            "created_at": row[3],
        }
        for row in list_assignments_by_teacher(teacher_name)
    ]


@app.get(
    "/assignments/student/{student_name}",
    response_model=list[StudentAssignmentResponse],
)
def student_assignments(student_name: str):
    if get_student_profile(student_name) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student '{student_name}' was not found",
        )

    return [
        {
            "id": row[0],
            "title": row[1],
            "subject": row[2],
            "created_at": row[3],
            "status": row[4],
        }
        for row in list_assignments_for_student(student_name)
    ]


@app.get(
    "/assignments/{assignment_id}/questions",
    response_model=list[AssignmentQuestionResponse],
)
def assignment_questions(assignment_id: int):
    if get_assignment(assignment_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assignment {assignment_id} was not found",
        )

    return [
        {
            "id": row[0],
            "question_order": row[1],
            "question": row[2],
            "max_points": row[4],
        }
        for row in get_assignment_questions(assignment_id)
    ]


@app.post(
    "/assignments/{assignment_id}/submit",
)
async def assignment_submit(
    assignment_id: int,
    request: AssignmentSubmissionRequest,
):
    assignment = get_assignment(assignment_id)
    if assignment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assignment {assignment_id} was not found",
        )

    student = get_student_profile(request.student_name)
    if student is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student '{request.student_name}' was not found",
        )
    if (
        student["teacher_name"] is None
        or student["teacher_name"].strip().lower()
        != assignment[1].strip().lower()
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This assignment is not available to this student",
        )

    question_rows = get_assignment_questions(assignment_id)
    if len(request.answers) != len(question_rows):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Expected {len(question_rows)} answers, "
                f"received {len(request.answers)}"
            ),
        )

    questions = [
        {
            "question": row[2],
            "model_answer": row[3],
            "max_points": row[4],
        }
        for row in question_rows
    ]
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI grading is unavailable because GEMINI_API_KEY is not set",
        )

    try:
        grading = await asyncio.to_thread(
            _grade_assignment,
            assignment[2],
            questions,
            request.answers,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AI grading returned an invalid result: {error}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI grading failed: {error}",
        ) from error

    save_assignment_submission(
        assignment_id,
        request.student_name,
        request.answers,
        grading,
    )
    return {"success": True, **grading}


@app.get(
    "/assignments/{assignment_id}/submission/{student_name}",
    response_model=AssignmentSubmissionResponse,
)
def assignment_submission(assignment_id: int, student_name: str):
    if get_assignment(assignment_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assignment {assignment_id} was not found",
        )

    submission = get_assignment_submission(assignment_id, student_name)
    if submission is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No submission was found for this student and assignment",
        )
    return submission
