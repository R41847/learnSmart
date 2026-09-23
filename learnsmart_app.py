import streamlit as st
import pandas as pd
import joblib
import sqlite3
import sys
import json
import io
from pathlib import Path

from gtts import gTTS
from database import (
    authenticate_user,
    create_tables,
    create_user,
    get_student_performance_trend,
    get_student_profile,
    get_student_records,
    get_connection
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="LearnSmart AI",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "student_performance_data (1).csv"
MODEL_FILE = BASE_DIR / "learnsmart_model.joblib"
RAG_DIR = BASE_DIR / "Rag"

create_tables()


# ============================================================
# LOAD DATA + MODEL
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


df = load_data()
model = load_model()


# ============================================================
# PERFORMANCE LEVEL
# ============================================================

def get_performance_level(grade):
    if grade in ["A", "B"]:
        return "Good"
    elif grade == "C":
        return "Average"
    else:
        return "At Risk"


df["performance_level"] = df["grade"].apply(get_performance_level)


# ============================================================
# DEMO RELATIONAL DATA
# ============================================================
#
# IMPORTANT:
# The original 10,000 records are kept for ML.
# The application UI uses only these 15 realistic demo students.
#
# Relationships are FIXED:
# Parent -> Children
# Teacher -> Students
# School -> Classes -> Students
#
# No modulo/random relationships.
# ============================================================

DEMO_STUDENTS = [
    {
        "student_name": "Ahmed Ali",
        "teacher_name": "Ms. Sara Hassan",
        "class_name": "5A",
        "school_name": "Nile Future School",
        "parent_name": "Ahmed's Mother"
    },
    {
        "student_name": "Omar Mohamed",
        "teacher_name": "Ms. Sara Hassan",
        "class_name": "5A",
        "school_name": "Nile Future School",
        "parent_name": "Omar's Father"
    },
    {
        "student_name": "Youssef Hassan",
        "teacher_name": "Ms. Sara Hassan",
        "class_name": "5A",
        "school_name": "Nile Future School",
        "parent_name": "Youssef's Mother"
    },
    {
        "student_name": "Jana Ahmed",
        "teacher_name": "Ms. Sara Hassan",
        "class_name": "5A",
        "school_name": "Nile Future School",
        "parent_name": "Jana's Mother"
    },
    {
        "student_name": "Mariam Ali",
        "teacher_name": "Mr. Ahmed Khaled",
        "class_name": "5B",
        "school_name": "Nile Future School",
        "parent_name": "Ahmed's Mother"
    },
    {
        "student_name": "Adam Mahmoud",
        "teacher_name": "Mr. Ahmed Khaled",
        "class_name": "5B",
        "school_name": "Nile Future School",
        "parent_name": "Adam's Father"
    },
    {
        "student_name": "Laila Mohamed",
        "teacher_name": "Mr. Ahmed Khaled",
        "class_name": "5B",
        "school_name": "Nile Future School",
        "parent_name": "Youssef's Mother"
    },
    {
        "student_name": "Malak Hassan",
        "teacher_name": "Ms. Mariam Ali",
        "class_name": "5C",
        "school_name": "Nile Future School",
        "parent_name": "Omar's Father"
    },
    {
        "student_name": "Yassin Mahmoud",
        "teacher_name": "Ms. Mariam Ali",
        "class_name": "5C",
        "school_name": "Nile Future School",
        "parent_name": "Yassin's Mother"
    },
    {
        "student_name": "Nour Ahmed",
        "teacher_name": "Ms. Mariam Ali",
        "class_name": "5C",
        "school_name": "Nile Future School",
        "parent_name": "Nour's Mother"
    },
    {
        "student_name": "Seif Khaled",
        "teacher_name": "Mr. Omar Hassan",
        "class_name": "6A",
        "school_name": "Al Noor Primary School",
        "parent_name": "Seif's Father"
    },
    {
        "student_name": "Lina Mostafa",
        "teacher_name": "Mr. Omar Hassan",
        "class_name": "6A",
        "school_name": "Al Noor Primary School",
        "parent_name": "Lina's Mother"
    },
    {
        "student_name": "Omar Hassan",
        "teacher_name": "Mr. Nada Mohamed",
        "class_name": "6B",
        "school_name": "Al Noor Primary School",
        "parent_name": "Omar Hassan's Mother"
    },
    {
        "student_name": "Maya Ahmed",
        "teacher_name": "Mr. Nada Mohamed",
        "class_name": "6B",
        "school_name": "Al Noor Primary School",
        "parent_name": "Maya's Mother"
    },
    {
        "student_name": "Kareem Ali",
        "teacher_name": "Mr. Nada Mohamed",
        "class_name": "6B",
        "school_name": "Al Noor Primary School",
        "parent_name": "Kareem's Father"
    }
]


# ============================================================
# CREATE DEMO DATASET
# ============================================================

demo_df = df.head(len(DEMO_STUDENTS)).copy()

demo_relationships = pd.DataFrame(DEMO_STUDENTS)

demo_df = demo_df.reset_index(drop=True)
demo_relationships = demo_relationships.reset_index(drop=True)

demo_df["student_name"] = demo_relationships["student_name"]
demo_df["teacher_name"] = demo_relationships["teacher_name"]
demo_df["class_name"] = demo_relationships["class_name"]
demo_df["school_name"] = demo_relationships["school_name"]
demo_df["parent_name"] = demo_relationships["parent_name"]


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = None

if "user_name" not in st.session_state:
    st.session_state.user_name = None

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "selected_student_name" not in st.session_state:
    st.session_state.selected_student_name = None

if "accessibility_mode" not in st.session_state:
    st.session_state.accessibility_mode = False

if "language" not in st.session_state:
    st.session_state.language = "en"

if "learning_plan" not in st.session_state:
    st.session_state.learning_plan = None

if "learning_plan_student" not in st.session_state:
    st.session_state.learning_plan_student = None

if "learning_plan_language" not in st.session_state:
    st.session_state.learning_plan_language = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def reset_session():
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.user_name = None
    st.session_state.user_id = None
    st.session_state.selected_student_name = None


def get_student(student_name):
    result = demo_df[
        demo_df["student_name"] == student_name
    ]

    if not result.empty:
        return result.iloc[0]

    return get_student_profile(student_name)


def get_current_students():
    """Use relational records when available, retaining demo data as fallback."""
    database_role = {
        "School Admin": "admin"
    }.get(role, role.lower())

    records = get_student_records(
        st.session_state.user_id,
        database_role,
        user_name
    )
    if records:
        return pd.DataFrame(records)

    if role == "Teacher":
        return demo_df[
            demo_df["teacher_name"] == user_name
        ].copy()
    if role == "Parent":
        return demo_df[
            demo_df["parent_name"] == user_name
        ].copy()
    if role == "School Admin":
        return demo_df[
            demo_df["school_name"] == user_name
        ].copy()
    return demo_df[
        demo_df["student_name"] == user_name
    ].copy()


def render_performance_trend(student_name, key_suffix):
    trend_data = get_student_performance_trend(
        student_name=student_name
    )
    points = trend_data["points"]

    st.subheader("📈 Performance Trend")
    if not points:
        st.info("No performance history or assignment submissions yet.")
        return trend_data

    chart_data = pd.DataFrame(
        {
            "Date": [point["date"] for point in points],
            "Score": [point["score"] for point in points]
        }
    ).set_index("Date")
    st.line_chart(chart_data, y="Score")

    recent = trend_data["recent_average"]
    earlier = trend_data["earlier_average"]
    if earlier is None:
        st.info(
            f"Trend: **{trend_data['trend']}** "
            f"(current average: {recent:.1f})"
        )
    else:
        st.info(
            f"Trend: **{trend_data['trend']}** — "
            f"earlier average: **{earlier:.1f}**, "
            f"recent average: **{recent:.1f}**"
        )
    return trend_data


def render_read_aloud_button(text, key):
    if not text or not text.strip():
        return

    if st.button("🔊 Read Aloud", key=key):
        try:
            audio_buffer = io.BytesIO()
            gTTS(text=text, lang="en").write_to_fp(audio_buffer)
            st.audio(
                audio_buffer.getvalue(),
                format="audio/mp3",
                autoplay=True
            )
        except Exception as error:
            st.error("Could not generate audio for this text.")
            st.exception(error)


def student_card(student):
    st.markdown(
        f"""
        <div style="
            padding:20px;
            border-radius:15px;
            border:1px solid #ddd;
            margin-bottom:15px;
            background-color:#fafafa;
        ">
            <h3>👨‍🎓 {student['student_name']}</h3>
            <p><b>Class:</b> {student['class_name']}</p>
            <p><b>Teacher:</b> {student['teacher_name']}</p>
            <p><b>School:</b> {student['school_name']}</p>
            <p><b>Performance:</b> {student['performance_level']}</p>
            <p><b>Overall Score:</b> {student['overall_score']:.1f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def performance_badge(level):
    if level == "Good":
        return "🟢 Good"
    elif level == "Average":
        return "🟡 Average"
    else:
        return "🔴 At Risk"


def get_teacher_assignments(teacher_name):
    conn = get_connection()
    assignments = conn.execute(
        """
        SELECT id, title, subject, instructions, created_at
        FROM assignments
        WHERE teacher_name = ?
        ORDER BY created_at DESC, id DESC
        """,
        (teacher_name,)
    ).fetchall()
    conn.close()
    return assignments


def get_assignment_questions(assignment_id):
    conn = get_connection()
    questions = conn.execute(
        """
        SELECT id, question_order, question, model_answer, max_points
        FROM assignment_questions
        WHERE assignment_id = ?
        ORDER BY question_order
        """,
        (assignment_id,)
    ).fetchall()
    conn.close()
    return questions


def get_student_submission(assignment_id, student_name):
    conn = get_connection()
    submission = conn.execute(
        """
        SELECT answers_json, grading_json, earned_points, max_points, percentage
        FROM assignment_submissions
        WHERE assignment_id = ? AND student_name = ?
        """,
        (assignment_id, student_name)
    ).fetchone()
    conn.close()
    return submission


def save_assignment(teacher_name, title, subject, instructions, questions):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO assignments (teacher_name, title, subject, instructions)
        VALUES (?, ?, ?, ?)
        """,
        (teacher_name, title, subject, instructions)
    )
    assignment_id = cursor.lastrowid
    cursor.executemany(
        """
        INSERT INTO assignment_questions
            (assignment_id, question_order, question, model_answer, max_points)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                assignment_id,
                index,
                question["question"],
                question["model_answer"],
                question["max_points"]
            )
            for index, question in enumerate(questions, 1)
        ]
    )
    conn.commit()
    conn.close()


def save_submission(
    assignment_id,
    student_name,
    answers,
    grading
):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO assignment_submissions
            (
                assignment_id,
                student_name,
                answers_json,
                grading_json,
                earned_points,
                max_points,
                percentage
            )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            assignment_id,
            student_name,
            json.dumps(answers),
            json.dumps(grading),
            grading["earned_points"],
            grading["max_points"],
            grading["percentage"]
        )
    )
    conn.commit()
    conn.close()


# ============================================================
# ROLE SELECTION SCREEN
# ============================================================

if not st.session_state.logged_in:

    st.title("🎓 LearnSmart AI")

    st.markdown(
        """
        ### AI-Powered Adaptive Learning Platform

        LearnSmart AI helps teachers, parents, schools,
        and students understand learning performance and
        provide personalized learning support.
        """
    )

    st.info(
        "Accounts are separate from the migrated student performance dataset. "
        "The existing dashboards still use the built-in demo relationships "
        "until those records are explicitly linked to user accounts."
    )

    st.divider()

    st.subheader("👋 Welcome to LearnSmart AI")

    view = st.radio(
        "Account access",
        ["Login", "Sign Up"],
        horizontal=True
    )

    if view == "Login":
        with st.form("login_form"):
            email = st.text_input("Email", autocomplete="email")
            password = st.text_input(
                "Password",
                type="password",
                autocomplete="current-password"
            )
            submitted = st.form_submit_button(
                "Login",
                use_container_width=True,
                type="primary"
            )

        if submitted:
            user = authenticate_user(email, password)
            if user is None:
                st.error("Invalid email or password.")
            else:
                role = "School Admin" if user["role"] == "admin" else user["role"].title()
                st.session_state.logged_in = True
                st.session_state.user_id = user["id"]
                st.session_state.role = role
                st.session_state.user_name = user["name"]
                if role == "Student":
                    st.session_state.selected_student_name = user["name"]
                st.rerun()
    else:
        with st.form("signup_form"):
            role = st.selectbox(
                "Choose your role",
                ["Teacher", "Parent", "School Admin", "Student"]
            )
            name = st.text_input("Name")
            email = st.text_input("Email", autocomplete="email")
            password = st.text_input(
                "Password",
                type="password",
                autocomplete="new-password"
            )
            confirm_password = st.text_input(
                "Confirm password",
                type="password",
                autocomplete="new-password"
            )
            submitted = st.form_submit_button(
                "Create account",
                use_container_width=True,
                type="primary"
            )

        if submitted:
            if not name.strip() or not email.strip() or not password:
                st.error("Name, email, and password are required.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                db_role = "admin" if role == "School Admin" else role.lower()
                try:
                    create_user(name, email, password, db_role)
                except sqlite3.IntegrityError:
                    st.error("An account with that email already exists.")
                else:
                    st.success("Account created. Switch to Login to continue.")

    st.stop()


# ============================================================
# CURRENT USER
# ============================================================

role = st.session_state.role
user_name = st.session_state.user_name


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎓 LearnSmart AI")

st.sidebar.markdown(
    f"""
    **Role:** {role}

    **Account:**  
    {user_name}
    """
)

st.sidebar.divider()


if role == "Teacher":

    pages = [
        "🏠 Home",
        "👨‍🎓 Students",
        "📝 Assignments",
        "👩‍🏫 Teacher Dashboard",
        "🤖 AI Assistant"
    ]

elif role == "Parent":

    pages = [
        "🏠 Home",
        "👨‍👩‍👧 Parent Dashboard"
    ]

elif role == "School Admin":

    pages = [
        "🏠 Home",
        "🏫 School Dashboard"
    ]

else:

    pages = [
        "🏠 Home",
        "📚 My Learning",
        "📝 Assignments"
    ]


page = st.sidebar.radio(
    "Navigation",
    pages
)

st.sidebar.divider()

st.session_state.accessibility_mode = st.sidebar.checkbox(
    "♿ High contrast / larger text",
    value=st.session_state.accessibility_mode
)

st.session_state.language = st.sidebar.radio(
    "🌐 Language",
    options=["en", "ar"],
    format_func=lambda value: (
        "English" if value == "en" else "العربية"
    ),
    horizontal=True
)

if st.session_state.accessibility_mode:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: #000000;
            color: #ffffff;
        }

        [data-testid="stSidebar"] {
            background: #000000;
            color: #ffffff;
        }

        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] span,
        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3 {
            color: #ffffff;
            font-size: 1.15rem;
        }

        [data-testid="stAppViewContainer"] button,
        [data-testid="stAppViewContainer"] input,
        [data-testid="stAppViewContainer"] textarea {
            font-size: 1.1rem;
            border: 2px solid #ffffff;
        }

        [data-testid="stAppViewContainer"] button {
            background: #000000;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

if st.sidebar.button(
    "🔄 Change Role",
    use_container_width=True
):
    reset_session()
    st.rerun()


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title("🎓 LearnSmart AI")

    st.subheader(
        f"Welcome, {user_name} 👋"
    )

    st.write(
        "AI-powered personalized learning for students, "
        "teachers, parents, and schools."
    )

    st.divider()

    current_students = get_current_students()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "👨‍🎓 My Students",
            len(current_students)
        )

    with col2:
        st.metric(
            "📊 Average Score",
            f"{current_students['overall_score'].mean():.1f}"
        )

    with col3:
        at_risk = (
            current_students["performance_level"] == "At Risk"
        ).sum()

        st.metric(
            "⚠️ At Risk",
            at_risk
        )

    with col4:
        st.metric(
            "📅 Avg Attendance",
            f"{current_students['attendance_percentage'].mean():.1f}%"
        )

    st.divider()

    if role == "Teacher":

        st.info(
            "👩‍🏫 You can view your assigned students, "
            "analyze their performance, and generate "
            "AI-powered personalized learning plans."
        )

    elif role == "Parent":

        st.info(
            "👨‍👩‍👧 You can view your children's academic "
            "performance and learning progress."
        )

    elif role == "School Admin":

        st.info(
            "🏫 You can monitor students and performance "
            "across your school."
        )

    else:

        st.info(
            "📚 You can view your learning performance "
            "and personalized learning information."
        )


# ============================================================
# STUDENTS PAGE - TEACHER ONLY
# ============================================================

elif page == "👨‍🎓 Students":

    st.title("👨‍🎓 My Students")

    teacher_students = get_current_students()

    st.write(
        f"You are viewing the students assigned to "
        f"**{user_name}**."
    )

    search = st.text_input(
        "🔍 Search student",
        placeholder="Enter student name..."
    )

    if search:

        teacher_students = teacher_students[
            teacher_students["student_name"]
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]

    st.divider()

    if teacher_students.empty:

        st.warning("No students found.")

    else:

        for _, student in teacher_students.iterrows():

            with st.expander(
                f"👨‍🎓 {student['student_name']} — "
                f"{performance_badge(student['performance_level'])}"
            ):

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Overall Score",
                        f"{student['overall_score']:.1f}"
                    )

                with col2:
                    st.metric(
                        "Attendance",
                        f"{student['attendance_percentage']:.1f}%"
                    )

                with col3:
                    st.metric(
                        "Study Hours",
                        f"{student['study_hours_per_day']:.1f}"
                    )

                st.write(
                    f"**Class:** {student['class_name']}"
                )

                st.write(
                    f"**Parent:** {student['parent_name']}"
                )

                st.write(
                    f"**School:** {student['school_name']}"
                )


# ============================================================
# TEACHER - ASSIGNMENTS
# ============================================================

elif page == "📝 Assignments" and role == "Teacher":

    st.title("📝 Assignments")
    st.write(
        "Create assignments with model answers. Students will receive "
        "automatic AI grading and written feedback after submission."
    )

    st.subheader("Create an assignment")

    with st.form("create_assignment_form"):
        title = st.text_input("Assignment title")
        subject = st.text_input("Subject")
        instructions = st.text_area("Instructions (optional)")
        question_count = st.number_input(
            "Number of questions",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )

        questions = []
        for index in range(int(question_count)):
            st.markdown(f"**Question {index + 1}**")
            question = st.text_area(
                "Question",
                key=f"new_assignment_question_{index}"
            )
            model_answer = st.text_area(
                "Model answer",
                key=f"new_assignment_model_answer_{index}"
            )
            max_points = st.number_input(
                "Maximum points",
                min_value=0.5,
                value=1.0,
                step=0.5,
                key=f"new_assignment_points_{index}"
            )
            questions.append(
                {
                    "question": question.strip(),
                    "model_answer": model_answer.strip(),
                    "max_points": float(max_points)
                }
            )

        create_clicked = st.form_submit_button(
            "Create Assignment",
            use_container_width=True,
            type="primary"
        )

    if create_clicked:
        if not title.strip():
            st.error("Please enter an assignment title.")
        elif any(
            not item["question"] or not item["model_answer"]
            for item in questions
        ):
            st.error("Every question must include a question and model answer.")
        else:
            save_assignment(
                user_name,
                title.strip(),
                subject.strip(),
                instructions.strip(),
                questions
            )
            st.success("Assignment created successfully.")
            st.rerun()

    st.divider()
    st.subheader("Your assignments")

    assignments = get_teacher_assignments(user_name)
    if not assignments:
        st.info("You have not created any assignments yet.")
    else:
        for assignment in assignments:
            assignment_id, assignment_title, subject, instructions, created_at = assignment
            with st.expander(
                f"{assignment_title}"
                + (f" — {subject}" if subject else "")
            ):
                st.caption(f"Created: {created_at}")
                if instructions:
                    st.write(instructions)
                assignment_questions = get_assignment_questions(assignment_id)
                for item in assignment_questions:
                    st.markdown(
                        f"**{item[1]}. {item[2]}** "
                        f"({item[4]:g} points)"
                    )
                    st.write(f"Model answer: {item[3]}")


# ============================================================
# TEACHER DASHBOARD
# ============================================================

elif page == "👩‍🏫 Teacher Dashboard":

    st.title("👩‍🏫 Teacher Dashboard")

    teacher_students = get_current_students()

    st.subheader(
        f"Students of {user_name}"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Students",
            len(teacher_students)
        )

    with col2:
        st.metric(
            "Average Score",
            f"{teacher_students['overall_score'].mean():.1f}"
        )

    with col3:
        st.metric(
            "Average Attendance",
            f"{teacher_students['attendance_percentage'].mean():.1f}%"
        )

    with col4:
        at_risk = (
            teacher_students["performance_level"] == "At Risk"
        ).sum()

        st.metric(
            "At Risk",
            at_risk
        )

    st.divider()

    st.subheader("📊 Class Performance")

    display_columns = [
        "student_name",
        "class_name",
        "overall_score",
        "attendance_percentage",
        "performance_level"
    ]

    table = teacher_students[display_columns].copy()

    table.columns = [
        "Student",
        "Class",
        "Overall Score",
        "Attendance %",
        "Performance"
    ]

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("📈 Performance Distribution")

    performance_counts = (
        teacher_students["performance_level"]
        .value_counts()
    )

    st.bar_chart(performance_counts)

    if not teacher_students.empty:
        selected_trend_student = st.selectbox(
            "Select a student to view performance trend",
            teacher_students["student_name"].tolist(),
            key="teacher_dashboard_trend_student"
        )
        render_performance_trend(
            selected_trend_student,
            "teacher_dashboard"
        )


# ============================================================
# PARENT DASHBOARD
# ============================================================

elif page == "👨‍👩‍👧 Parent Dashboard":

    st.title("👨‍👩‍👧 Parent Dashboard")

    parent_children = get_current_students()

    st.success(
        f"Welcome! Showing only the children linked "
        f"to **{user_name}**."
    )

    st.subheader(
        f"👨‍👩‍👧 My Children ({len(parent_children)})"
    )

    if parent_children.empty:

        st.warning(
            "No children are linked to this account."
        )

    else:

        for _, child in parent_children.iterrows():

            st.markdown("---")

            st.subheader(
                f"👨‍🎓 {child['student_name']}"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Overall Score",
                    f"{child['overall_score']:.1f}"
                )

            with col2:
                st.metric(
                    "Attendance",
                    f"{child['attendance_percentage']:.1f}%"
                )

            with col3:
                st.metric(
                    "Study Hours",
                    f"{child['study_hours_per_day']:.1f}"
                )

            with col4:
                st.metric(
                    "Performance",
                    child["performance_level"]
                )

            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    f"🏫 **School:** {child['school_name']}"
                )

                st.write(
                    f"📚 **Class:** {child['class_name']}"
                )

            with col2:

                st.write(
                    f"👩‍🏫 **Teacher:** {child['teacher_name']}"
                )

                st.write(
                    f"📊 **Grade:** {child['grade']}"
                )


# ============================================================
# SCHOOL DASHBOARD
# ============================================================

elif page == "🏫 School Dashboard":

    st.title("🏫 School Dashboard")

    school_students = demo_df[
        demo_df["school_name"] == user_name
    ].copy()

    st.success(
        f"Showing students from **{user_name}** only."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "👨‍🎓 Students",
            len(school_students)
        )

    with col2:
        st.metric(
            "📊 Average Score",
            f"{school_students['overall_score'].mean():.1f}"
        )

    with col3:
        st.metric(
            "📅 Avg Attendance",
            f"{school_students['attendance_percentage'].mean():.1f}%"
        )

    with col4:

        at_risk = (
            school_students["performance_level"] == "At Risk"
        ).sum()

        st.metric(
            "⚠️ At Risk",
            at_risk
        )

    st.divider()

    st.subheader("🏫 School Students")

    display_columns = [
        "student_name",
        "class_name",
        "teacher_name",
        "overall_score",
        "attendance_percentage",
        "performance_level"
    ]

    table = school_students[display_columns].copy()

    table.columns = [
        "Student",
        "Class",
        "Teacher",
        "Overall Score",
        "Attendance %",
        "Performance"
    ]

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("📚 Classes")

    class_summary = (
        school_students
        .groupby("class_name")
        .agg(
            Students=("student_name", "count"),
            Average_Score=("overall_score", "mean"),
            Attendance=("attendance_percentage", "mean")
        )
        .reset_index()
    )

    class_summary["Average_Score"] = (
        class_summary["Average_Score"].round(1)
    )

    class_summary["Attendance"] = (
        class_summary["Attendance"].round(1)
    )

    class_summary.columns = [
        "Class",
        "Students",
        "Average Score",
        "Attendance %"
    ]

    st.dataframe(
        class_summary,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# STUDENT - MY LEARNING
# ============================================================

elif page == "📚 My Learning":

    st.title("📚 My Learning")

    student = get_student(user_name)

    if student is None:

        st.error("Student profile not found.")

    else:

        st.subheader(
            f"Welcome, {student['student_name']} 👋"
        )

        st.write(
            "Here is your current learning profile."
        )

        st.divider()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Overall Score",
                f"{student['overall_score']:.1f}"
            )

        with col2:
            st.metric(
                "Attendance",
                f"{student['attendance_percentage']:.1f}%"
            )

        with col3:
            st.metric(
                "Study Hours",
                f"{student['study_hours_per_day']:.1f}"
            )

        with col4:
            st.metric(
                "Performance",
                student["performance_level"]
            )

        st.divider()

        st.subheader("📊 Academic Performance")

        scores = pd.DataFrame(
            {
                "Assessment": [
                    "Assignment",
                    "Midterm",
                    "Final Exam",
                    "Participation"
                ],
                "Score": [
                    student["assignment_score"],
                    student["midterm_score"],
                    student["final_exam_score"],
                    student["participation_score"]
                ]
            }
        )

        st.bar_chart(
            scores.set_index("Assessment")
        )

        st.divider()

        render_performance_trend(
            student["student_name"],
            "student_learning"
        )

        st.divider()

        st.write(
            f"🏫 **School:** {student['school_name']}"
        )

        st.write(
            f"📚 **Class:** {student['class_name']}"
        )

        st.write(
            f"👩‍🏫 **Teacher:** {student['teacher_name']}"
        )


# ============================================================
# STUDENT - ASSIGNMENTS
# ============================================================

elif page == "📝 Assignments" and role == "Student":

    st.title("📝 Assignments")

    student = get_student(user_name)
    if student is None:
        st.error("Student profile not found.")
    else:
        assignments = get_teacher_assignments(student["teacher_name"])

        if not assignments:
            st.info("There are no assignments from your teacher yet.")
        else:
            for assignment in assignments:
                assignment_id, title, subject, instructions, created_at = assignment
                assignment_questions = get_assignment_questions(assignment_id)
                submission = get_student_submission(assignment_id, user_name)

                with st.expander(
                    f"{title}"
                    + (f" — {subject}" if subject else ""),
                    expanded=submission is None
                ):
                    st.caption(f"Created: {created_at}")
                    if instructions:
                        st.write(instructions)

                    if submission is not None:
                        (
                            answers_json,
                            grading_json,
                            earned_points,
                            max_points,
                            percentage
                        ) = submission
                        answers = json.loads(answers_json)
                        grading = json.loads(grading_json)

                        st.success(
                            f"Score: {earned_points:g} / {max_points:g} "
                            f"({percentage:.1f}%)"
                        )

                        feedback_parts = []

                        for index, item in enumerate(
                            assignment_questions
                        ):
                            st.markdown(f"**{index + 1}. {item[2]}**")
                            st.write(f"Your answer: {answers[index]}")
                            feedback = grading["questions"][index]
                            feedback_parts.append(
                                f"Question {index + 1}: "
                                f"{feedback['points_awarded']:g} out of "
                                f"{feedback['max_points']:g} points. "
                                f"{feedback['feedback']}"
                            )
                            st.info(
                                f"**{feedback['points_awarded']:g} / "
                                f"{feedback['max_points']:g} points** — "
                                f"{feedback['feedback']}"
                            )

                        if grading.get("overall_feedback"):
                            feedback_parts.append(
                                f"Overall feedback: "
                                f"{grading['overall_feedback']}"
                            )
                            st.write(
                                f"**Overall feedback:** "
                                f"{grading['overall_feedback']}"
                            )

                        render_read_aloud_button(
                            "\n".join(feedback_parts),
                            key=f"read_feedback_{assignment_id}"
                        )
                    else:
                        with st.form(f"submit_assignment_{assignment_id}"):
                            answers = []
                            for index, item in enumerate(
                                assignment_questions
                            ):
                                st.markdown(
                                    f"**{index + 1}. {item[2]}** "
                                    f"({item[4]:g} points)"
                                )
                                answers.append(
                                    st.text_area(
                                        "Your answer",
                                        key=f"answer_{assignment_id}_{index}"
                                    )
                                )

                            submit_clicked = st.form_submit_button(
                                "Submit for AI Grading",
                                use_container_width=True,
                                type="primary"
                            )

                        if submit_clicked:
                            if any(not answer.strip() for answer in answers):
                                st.error("Please answer every question before submitting.")
                            else:
                                try:
                                    rag_path = str(RAG_DIR)
                                    if rag_path not in sys.path:
                                        sys.path.insert(0, rag_path)

                                    from genai import grade_assignment

                                    questions_for_grading = [
                                        {
                                            "question": item[2],
                                            "model_answer": item[3],
                                            "max_points": item[4]
                                        }
                                        for item in assignment_questions
                                    ]

                                    with st.spinner(
                                        "AI is grading your assignment..."
                                    ):
                                        grading = grade_assignment(
                                            title,
                                            questions_for_grading,
                                            answers
                                        )

                                    save_submission(
                                        assignment_id,
                                        user_name,
                                        answers,
                                        grading
                                    )
                                    st.success(
                                        "Assignment graded successfully."
                                    )
                                    st.rerun()
                                except Exception as error:
                                    st.error(
                                        "Could not grade the assignment."
                                    )
                                    st.exception(error)


# ============================================================
# AI ASSISTANT
# ============================================================

elif page == "🤖 AI Assistant":

    st.title("🤖 LearnSmart AI Assistant")

    st.write(
        "Generate a personalized learning plan using "
        "ML + RAG + Gemini."
    )

    # --------------------------------------------------------
    # ONLY TEACHER'S STUDENTS
    # --------------------------------------------------------

    teacher_students = demo_df[
        demo_df["teacher_name"] == user_name
    ].copy()

    st.info(
        f"👩‍🏫 You are logged in as **{user_name}**. "
        "You can generate plans only for your assigned students."
    )

    selected_student_name = st.selectbox(
        "👨‍🎓 Select student",
        teacher_students["student_name"].tolist()
    )

    student = get_student(selected_student_name)

    if student is None:

        st.error("Student not found.")

    else:

        st.divider()

        performance_trend = get_student_performance_trend(
            student_name=selected_student_name
        )

        # ----------------------------------------------------
        # STUDENT SUMMARY
        # ----------------------------------------------------

        st.subheader(
            f"👨‍🎓 {student['student_name']}"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Performance",
                student["performance_level"]
            )

        with col2:
            st.metric(
                "Overall Score",
                f"{student['overall_score']:.1f}"
            )

        with col3:
            st.metric(
                "Attendance",
                f"{student['attendance_percentage']:.1f}%"
            )

        with col4:
            st.metric(
                "Study Hours",
                f"{student['study_hours_per_day']:.1f}"
            )

        st.write(
            f"🏫 **School:** {student['school_name']}"
        )

        st.write(
            f"📚 **Class:** {student['class_name']}"
        )

        st.write(
            f"👩‍🏫 **Teacher:** {student['teacher_name']}"
        )

        st.divider()

        # ----------------------------------------------------
        # STUDENT INPUTS
        # ----------------------------------------------------

        st.subheader("🧠 Student Learning Profile")

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Student Age",
                min_value=5,
                max_value=18,
                value=10
            )

            learning_style = st.selectbox(
                "Learning Style",
                [
                    "Visual",
                    "Auditory",
                    "Reading/Writing",
                    "Kinesthetic"
                ]
            )

        with col2:

            preferred_topic = st.selectbox(
                "Preferred Topic",
                [
                    "Math",
                    "Science",
                    "English",
                    "Computer Science",
                    "General"
                ]
            )

        # ----------------------------------------------------
        # AUTOMATIC WEAK AREA
        # ----------------------------------------------------

        score_columns = {
            "Assignment": student["assignment_score"],
            "Midterm": student["midterm_score"],
            "Final Exam": student["final_exam_score"],
            "Participation": student["participation_score"]
        }

        weakest_area = min(
            score_columns,
            key=score_columns.get
        )

        weakest_score = score_columns[weakest_area]

        st.info(
            f"🎯 Automatically detected weakest area: "
            f"**{weakest_area} ({weakest_score:.1f})**"
        )

        st.divider()

        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        if st.button(
            "✨ Generate Personalized Learning Plan",
            use_container_width=True,
            type="primary"
        ):

            student_profile = {
                "performance_level": student["performance_level"],
                "age": age,
                "education_level": "Primary",
                "learning_style": learning_style,
                "preferred_topics": preferred_topic,
                "weak_areas": f"Low {weakest_area} performance",
                "study_hours": student["study_hours_per_day"],
                "attendance": student["attendance_percentage"]
            }

            with st.spinner(
                "🔎 Retrieving learning resources and generating AI plan..."
            ):

                try:

                    # Add Rag directory to Python path
                    rag_path = str(RAG_DIR)

                    if rag_path not in sys.path:
                        sys.path.insert(0, rag_path)

                    from genai import generate_learning_plan

                    st.session_state.learning_plan = generate_learning_plan(
                        student_profile,
                        language=st.session_state.language,
                        performance_trend=performance_trend
                    )
                    st.session_state.learning_plan_student = (
                        selected_student_name
                    )
                    st.session_state.learning_plan_language = (
                        st.session_state.language
                    )

                    st.success(
                        "✅ Personalized learning plan generated!"
                    )

                except Exception as e:

                    st.error(
                        "❌ Could not generate the AI learning plan."
                    )

                    st.exception(e)

        if (
            st.session_state.learning_plan
            and st.session_state.learning_plan_student
            == selected_student_name
            and st.session_state.learning_plan_language
            == st.session_state.language
        ):
            st.markdown("## 🧠 AI Learning Plan")
            st.markdown(st.session_state.learning_plan)
            render_read_aloud_button(
                st.session_state.learning_plan,
                key="read_learning_plan"
            )
        elif (
            st.session_state.learning_plan
            and st.session_state.learning_plan_student
            == selected_student_name
        ):
            st.info(
                "Generate the learning plan again to view it in the "
                "selected language."
            )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "LearnSmart AI • Adaptive Learning Platform"
)