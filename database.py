import hashlib
import secrets
import sqlite3
from statistics import mean


DATABASE_NAME = "learnsmart.db"
PASSWORD_ITERATIONS = 100_000


def get_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def hash_password(password):
    """Return a salted password hash suitable for storing in the users table."""
    salt = secrets.token_bytes(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS
    )
    return f"{salt.hex()}${password_hash.hex()}"


def verify_password(password, stored_password):
    """Check a plaintext password against a stored salt and hash."""
    try:
        salt_hex, hash_hex = stored_password.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        expected_hash = bytes.fromhex(hash_hex)
    except (AttributeError, ValueError):
        return False

    actual_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS
    )
    return secrets.compare_digest(actual_hash, expected_hash)


def create_user(name, email, password, role):
    """Create a user with a normalized email and a salted password hash."""
    normalized_name = name.strip()
    normalized_email = email.strip().lower()
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO users (name, email, password, role)
            VALUES (?, ?, ?, ?)
            """,
            (
                normalized_name,
                normalized_email,
                hash_password(password),
                role
            )
        )
        user_id = cursor.lastrowid

        if role == "student":
            conn.execute(
                """
                UPDATE students
                SET user_id = ?
                WHERE user_id IS NULL
                AND lower(trim(student_name)) = lower(trim(?))
                """,
                (user_id, normalized_name)
            )
        elif role == "teacher":
            # Existing teacher rows currently have no teacher_name column.
            # Student.teacher_name remains the source for migrated assignments.
            conn.execute(
                """
                UPDATE teachers
                SET user_id = ?
                WHERE user_id IS NULL
                AND teacher_code = ?
                """,
                (user_id, normalized_name)
            )
        elif role == "parent":
            conn.execute(
                """
                INSERT INTO parents (user_id)
                VALUES (?)
                """,
                (user_id,)
            )

        conn.commit()
        return user_id
    finally:
        conn.close()


def authenticate_user(email, password):
    """Return the matching user, or None when credentials are invalid."""
    conn = get_connection()
    try:
        user = conn.execute(
            """
            SELECT id, name, email, role, password
            FROM users
            WHERE email = ?
            """,
            (email.strip().lower(),)
        ).fetchone()
    finally:
        conn.close()

    if user is None or not verify_password(password, user[4]):
        return None

    return {
        "id": user[0],
        "name": user[1],
        "email": user[2],
        "role": user[3]
    }


def get_student_records(user_id, role, name):
    """Return dashboard-ready student records from the relational database."""
    filters = []
    parameters = []

    if role == "student":
        filters.append("s.user_id = ?")
        parameters.append(user_id)
    elif role == "teacher":
        filters.append(
            """
            (
                lower(trim(s.teacher_name)) = lower(trim(?))
                OR s.teacher_id IN (
                    SELECT id FROM teachers WHERE user_id = ?
                )
            )
            """
        )
        parameters.extend([name, user_id])
    elif role == "parent":
        filters.append(
            """
            s.id IN (
                SELECT ps.student_id
                FROM parent_student ps
                JOIN parents p ON p.id = ps.parent_id
                WHERE p.user_id = ?
            )
            """
        )
        parameters.append(user_id)
    else:
        return []

    conn = get_connection()
    try:
        rows = conn.execute(
            f"""
            SELECT
                s.student_name,
                s.class_name,
                COALESCE(s.school_name, school.name) AS school_name,
                s.teacher_name,
                COALESCE(
                    (
                        SELECT group_concat(u.name, ', ')
                        FROM parent_student ps
                        JOIN parents p ON p.id = ps.parent_id
                        JOIN users u ON u.id = p.user_id
                        WHERE ps.student_id = s.id
                    ),
                    'Not linked'
                ) AS parent_name,
                h.overall_score,
                h.grade,
                h.attendance_percentage,
                h.study_hours_per_day,
                COALESCE(h.performance_level, 'At Risk') AS performance_level
            FROM students s
            LEFT JOIN schools school
                ON school.id = s.school_id
            LEFT JOIN student_history h
                ON h.id = (
                    SELECT latest.id
                    FROM student_history latest
                    WHERE latest.student_id = s.id
                    ORDER BY latest.date DESC, latest.id DESC
                    LIMIT 1
                )
            WHERE {" AND ".join(filters)}
            ORDER BY s.student_name
            """,
            parameters
        ).fetchall()
    finally:
        conn.close()

    columns = [
        "student_name",
        "class_name",
        "school_name",
        "teacher_name",
        "parent_name",
        "overall_score",
        "grade",
        "attendance_percentage",
        "study_hours_per_day",
        "performance_level"
    ]
    return [dict(zip(columns, row)) for row in rows]


def get_students_by_teacher(teacher_name):
    """Return dashboard-ready students assigned to a teacher by name."""
    return get_student_records(None, "teacher", teacher_name)


def get_students_by_parent_email(parent_email):
    """Return profile-ready students linked to a parent email."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                s.student_name,
                s.class_name,
                COALESCE(s.school_name, school.name) AS school_name,
                s.teacher_name,
                h.overall_score,
                h.attendance_percentage,
                h.study_hours_per_day,
                h.assignment_score,
                h.midterm_score,
                h.final_exam_score,
                h.participation_score
            FROM parent_student ps
            JOIN parents p
                ON p.id = ps.parent_id
            JOIN users u
                ON u.id = p.user_id
            JOIN students s
                ON s.id = ps.student_id
            LEFT JOIN schools school
                ON school.id = s.school_id
            LEFT JOIN student_history h
                ON h.id = (
                    SELECT latest.id
                    FROM student_history latest
                    WHERE latest.student_id = s.id
                    ORDER BY latest.date DESC, latest.id DESC
                    LIMIT 1
                )
            WHERE lower(trim(u.email)) = lower(trim(?))
            ORDER BY s.student_name
            """,
            (parent_email,),
        ).fetchall()
    finally:
        conn.close()

    columns = [
        "student_name",
        "class_name",
        "school_name",
        "teacher_name",
        "overall_score",
        "attendance_percentage",
        "study_hours_per_day",
        "assignment_score",
        "midterm_score",
        "final_exam_score",
        "participation_score",
    ]
    return [dict(zip(columns, row)) for row in rows]


def get_student_performance_trend(student_name=None, student_id=None):
    """Combine history and assignment scores into a simple performance trend."""
    conn = get_connection()
    try:
        if student_id is None and student_name:
            student = conn.execute(
                """
                SELECT id, student_name
                FROM students
                WHERE lower(trim(student_name)) = lower(trim(?))
                """,
                (student_name,)
            ).fetchone()
            if student:
                student_id = student[0]
                student_name = student[1]

        history_rows = []
        if student_id is not None:
            history_rows = conn.execute(
                """
                SELECT date, overall_score
                FROM student_history
                WHERE student_id = ?
                AND overall_score IS NOT NULL
                """,
                (student_id,)
            ).fetchall()

        submission_rows = []
        if student_name:
            submission_rows = conn.execute(
                """
                SELECT submitted_at, percentage
                FROM assignment_submissions
                WHERE lower(trim(student_name)) = lower(trim(?))
                AND percentage IS NOT NULL
                """,
                (student_name,)
            ).fetchall()
    finally:
        conn.close()

    points = [
        {"date": str(date), "score": float(score), "source": "history"}
        for date, score in history_rows
    ]
    points.extend(
        {
            "date": str(date),
            "score": float(score),
            "source": "assignment"
        }
        for date, score in submission_rows
    )
    points.sort(key=lambda item: item["date"])

    if not points:
        return {
            "points": [],
            "trend": "stable",
            "recent_average": None,
            "earlier_average": None
        }

    if len(points) == 1:
        recent_average = points[0]["score"]
        earlier_average = None
        trend = "stable"
    else:
        split_index = max(1, len(points) // 2)
        earlier_scores = [
            point["score"] for point in points[:split_index]
        ]
        recent_scores = [
            point["score"] for point in points[split_index:]
        ]
        earlier_average = mean(earlier_scores)
        recent_average = mean(recent_scores)
        difference = recent_average - earlier_average

        if difference >= 5:
            trend = "improving"
        elif difference <= -5:
            trend = "declining"
        else:
            trend = "stable"

    return {
        "points": points,
        "trend": trend,
        "recent_average": recent_average,
        "earlier_average": earlier_average
    }


def get_student_profile(student_name):
    """Return a student's current profile from the relational database."""
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT
                s.student_name,
                s.class_name,
                COALESCE(s.school_name, school.name) AS school_name,
                s.teacher_name,
                h.overall_score,
                h.grade,
                h.performance_level,
                h.attendance_percentage,
                h.study_hours_per_day,
                h.assignment_score,
                h.midterm_score,
                h.final_exam_score,
                h.participation_score,
                h.sleep_hours
            FROM students s
            LEFT JOIN schools school
                ON school.id = s.school_id
            LEFT JOIN student_history h
                ON h.id = (
                    SELECT latest.id
                    FROM student_history latest
                    WHERE latest.student_id = s.id
                    ORDER BY latest.date DESC, latest.id DESC
                    LIMIT 1
                )
            WHERE lower(trim(s.student_name)) = lower(trim(?))
            """,
            (student_name,)
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    columns = [
        "student_name",
        "class_name",
        "school_name",
        "teacher_name",
        "overall_score",
        "grade",
        "performance_level",
        "attendance_percentage",
        "study_hours_per_day",
        "assignment_score",
        "midterm_score",
        "final_exam_score",
        "participation_score",
        "sleep_hours"
    ]
    return dict(zip(columns, row))


def create_tables():

    conn = get_connection()
    cursor = conn.cursor()

    # =====================================================
    # USERS
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
                CHECK(role IN ('admin', 'teacher', 'parent', 'student')),
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # =====================================================
    # SCHOOLS
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS schools (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            address TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # =====================================================
    # TEACHERS
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE,
            teacher_code TEXT UNIQUE NOT NULL,
            school_id INTEGER,
            subject TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (school_id) REFERENCES schools(id)
        )
    """)

    # =====================================================
    # STUDENTS
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_code TEXT UNIQUE NOT NULL,
            user_id INTEGER UNIQUE,
            student_name TEXT NOT NULL,
            gender TEXT,
            school_id INTEGER,
            school_name TEXT,
            class_name TEXT,
            teacher_id INTEGER,
            teacher_name TEXT,
            grade_level TEXT,
            date_of_birth TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (school_id) REFERENCES schools(id),
            FOREIGN KEY (teacher_id) REFERENCES teachers(id)
        )
    """)

    # =====================================================
    # PARENTS
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE,
            phone TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # =====================================================
    # PARENT - STUDENT
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parent_student (
            parent_id INTEGER,
            student_id INTEGER,

            PRIMARY KEY (parent_id, student_id),

            FOREIGN KEY (parent_id)
                REFERENCES parents(id)
                ON DELETE CASCADE,

            FOREIGN KEY (student_id)
                REFERENCES students(id)
                ON DELETE CASCADE
        )
    """)

    # =====================================================
    # SUBJECTS
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            grade_level TEXT
        )
    """)

    # =====================================================
    # ACADEMIC RECORDS
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS academic_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            student_id INTEGER NOT NULL,

            date TEXT,
            subject TEXT,
            score REAL,

            performance_level TEXT,

            teacher_note TEXT,

            FOREIGN KEY (student_id)
                REFERENCES students(id)
                ON DELETE CASCADE
        )
    """)

    # =====================================================
    # STUDENT HISTORY
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            student_id INTEGER NOT NULL,

            date TEXT NOT NULL,

            overall_score REAL,
            grade TEXT,
            performance_level TEXT,

            attendance_percentage REAL,
            study_hours_per_day REAL,

            assignment_score REAL,
            midterm_score REAL,
            final_exam_score REAL,

            participation_score REAL,
            sleep_hours REAL,

            FOREIGN KEY (student_id)
                REFERENCES students(id)
                ON DELETE CASCADE
        )
    """)

    # =====================================================
    # ATTENDANCE
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            student_id INTEGER NOT NULL,

            date TEXT NOT NULL,
            status TEXT NOT NULL,

            FOREIGN KEY (student_id)
                REFERENCES students(id)
                ON DELETE CASCADE
        )
    """)

    # =====================================================
    # TEACHER NOTES
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teacher_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            student_id INTEGER NOT NULL,

            teacher_name TEXT,
            note TEXT NOT NULL,
            date TEXT,

            FOREIGN KEY (student_id)
                REFERENCES students(id)
                ON DELETE CASCADE
        )
    """)

    # =====================================================
    # QUIZZES
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quizzes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            title TEXT NOT NULL,
            subject TEXT,
            topic TEXT,
            grade_level TEXT,

            difficulty TEXT,

            generated_by TEXT,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # =====================================================
    # QUIZ QUESTIONS
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            quiz_id INTEGER NOT NULL,

            question TEXT NOT NULL,

            option_a TEXT,
            option_b TEXT,
            option_c TEXT,
            option_d TEXT,

            correct_answer TEXT,

            explanation TEXT,

            FOREIGN KEY (quiz_id)
                REFERENCES quizzes(id)
                ON DELETE CASCADE
        )
    """)

    # =====================================================
    # QUIZ RESULTS
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            student_id INTEGER NOT NULL,
            quiz_id INTEGER,

            subject TEXT,
            topic TEXT,

            score REAL,
            total_questions INTEGER,

            percentage REAL,

            date TEXT,

            FOREIGN KEY (student_id)
                REFERENCES students(id)
                ON DELETE CASCADE,

            FOREIGN KEY (quiz_id)
                REFERENCES quizzes(id)
                ON DELETE SET NULL
        )
    """)

    # =====================================================
    # MENTAL GAMES
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mental_games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            title TEXT NOT NULL,
            game_type TEXT,

            subject TEXT,
            topic TEXT,

            grade_level TEXT,
            difficulty TEXT,

            description TEXT,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # =====================================================
    # GAME RESULTS
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS game_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            student_id INTEGER NOT NULL,
            game_id INTEGER,

            score REAL,
            accuracy REAL,

            duration_seconds INTEGER,

            date TEXT,

            FOREIGN KEY (student_id)
                REFERENCES students(id)
                ON DELETE CASCADE,

            FOREIGN KEY (game_id)
                REFERENCES mental_games(id)
                ON DELETE SET NULL
        )
    """)

    # =====================================================
    # AI INSIGHTS
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            student_id INTEGER NOT NULL,

            insight TEXT NOT NULL,

            risk_level TEXT,

            weak_topics TEXT,

            recommendations TEXT,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (student_id)
                REFERENCES students(id)
                ON DELETE CASCADE
        )
    """)

    # =====================================================
    # LEARNING PLANS
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learning_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            student_id INTEGER NOT NULL,

            plan TEXT NOT NULL,

            goals TEXT,
            weak_topics TEXT,

            duration_days INTEGER,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (student_id)
                REFERENCES students(id)
                ON DELETE CASCADE
        )
    """)

    # =====================================================
    # RAG SOURCES
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rag_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            title TEXT NOT NULL,
            source_type TEXT,

            file_path TEXT,
            url TEXT,

            description TEXT,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # =====================================================
    # RAG CHUNKS
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rag_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            source_id INTEGER NOT NULL,

            chunk_index INTEGER,

            content TEXT NOT NULL,

            topic TEXT,
            grade_level TEXT,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (source_id)
                REFERENCES rag_sources(id)
                ON DELETE CASCADE
        )
    """)

    # =====================================================
    # AI GENERATED CONTENT
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_generated_content (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            student_id INTEGER,

            content_type TEXT,

            prompt TEXT,

            generated_content TEXT,

            sources TEXT,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (student_id)
                REFERENCES students(id)
                ON DELETE CASCADE
        )
    """)

    # =====================================================
    # ASSIGNMENTS
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            teacher_name TEXT NOT NULL,
            title TEXT NOT NULL,
            subject TEXT,
            instructions TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assignment_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assignment_id INTEGER NOT NULL,
            question_order INTEGER NOT NULL,
            question TEXT NOT NULL,
            model_answer TEXT NOT NULL,
            max_points REAL NOT NULL DEFAULT 1,
            FOREIGN KEY (assignment_id)
                REFERENCES assignments(id)
                ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assignment_submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assignment_id INTEGER NOT NULL,
            student_name TEXT NOT NULL,
            answers_json TEXT NOT NULL,
            grading_json TEXT NOT NULL,
            earned_points REAL NOT NULL,
            max_points REAL NOT NULL,
            percentage REAL NOT NULL,
            submitted_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (assignment_id, student_name),
            FOREIGN KEY (assignment_id)
                REFERENCES assignments(id)
                ON DELETE CASCADE
        )
    """)

    # =====================================================
    # INDEXES
    # =====================================================

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_student_history_student
        ON student_history(student_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_academic_student
        ON academic_records(student_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_attendance_student
        ON attendance_records(student_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_quiz_results_student
        ON quiz_results(student_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_ai_insights_student
        ON ai_insights(student_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_learning_plans_student
        ON learning_plans(student_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_rag_chunks_source
        ON rag_chunks(source_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_assignments_teacher
        ON assignments(teacher_name)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_assignment_submissions_student
        ON assignment_submissions(student_name)
    """)

    conn.commit()
    conn.close()


if __name__ == "__main__":

    create_tables()

    print("======================================")
    print("LearnSmart AI Database")
    print("======================================")
    print("Database created successfully!")
    print("All tables are ready.")
    print("======================================")