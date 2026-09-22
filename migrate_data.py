import pandas as pd
from database import get_connection, create_tables


CSV_FILE = "student_performance_data (1).csv"


def assign_performance_level(grade):

    if grade in ["A", "B"]:
        return "Good"

    elif grade == "C":
        return "Average"

    return "At_Risk"


def migrate():

    print("Starting LearnSmart migration...")

    create_tables()

    df = pd.read_csv(CSV_FILE)

    conn = get_connection()
    cursor = conn.cursor()

    students_added = 0
    academic_added = 0
    history_added = 0

    # =====================================================
    # STUDENTS
    # =====================================================

    for _, row in df.iterrows():

        student_code = str(row["student_id"])

        performance_level = assign_performance_level(
            row["grade"]
        )

        # Check student
        cursor.execute(
            """
            SELECT id
            FROM students
            WHERE student_code = ?
            """,
            (student_code,)
        )

        result = cursor.fetchone()

        # -----------------------------------------------
        # Create student if not found
        # -----------------------------------------------

        if result is None:

            cursor.execute(
                """
                INSERT INTO students (
                    student_code,
                    student_name,
                    gender
                )
                VALUES (?, ?, ?)
                """,
                (
                    student_code,
                    f"Student {student_code}",
                    row["gender"]
                )
            )

            student_db_id = cursor.lastrowid

            students_added += 1

        else:

            student_db_id = result[0]

        # =================================================
        # ACADEMIC RECORDS
        # =================================================

        academic_data = [

            (
                "Study Hours",
                row["study_hours_per_day"]
            ),

            (
                "Attendance",
                row["attendance_percentage"]
            ),

            (
                "Assignment",
                row["assignment_score"]
            ),

            (
                "Midterm",
                row["midterm_score"]
            ),

            (
                "Final Exam",
                row["final_exam_score"]
            ),

            (
                "Participation",
                row["participation_score"]
            ),

            (
                "Sleep Hours",
                row["sleep_hours"]
            )
        ]

        for subject, score in academic_data:

            cursor.execute(
                """
                SELECT id
                FROM academic_records
                WHERE student_id = ?
                AND subject = ?
                """,
                (
                    student_db_id,
                    subject
                )
            )

            exists = cursor.fetchone()

            if exists is None:

                cursor.execute(
                    """
                    INSERT INTO academic_records (
                        student_id,
                        date,
                        subject,
                        score,
                        performance_level
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        student_db_id,
                        "2026-09-12",
                        subject,
                        float(score),
                        performance_level
                    )
                )

                academic_added += 1

        # =================================================
        # STUDENT HISTORY
        # =================================================

        cursor.execute(
            """
            SELECT id
            FROM student_history
            WHERE student_id = ?
            AND date = ?
            """,
            (
                student_db_id,
                "2026-09-12"
            )
        )

        history_exists = cursor.fetchone()

        if history_exists is None:

            cursor.execute(
                """
                INSERT INTO student_history (
                    student_id,
                    date,
                    overall_score,
                    grade,
                    performance_level,
                    attendance_percentage,
                    study_hours_per_day,
                    assignment_score,
                    midterm_score,
                    final_exam_score,
                    participation_score,
                    sleep_hours
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    student_db_id,
                    "2026-09-12",
                    float(row["overall_score"]),
                    row["grade"],
                    performance_level,
                    float(row["attendance_percentage"]),
                    float(row["study_hours_per_day"]),
                    float(row["assignment_score"]),
                    float(row["midterm_score"]),
                    float(row["final_exam_score"]),
                    float(row["participation_score"]),
                    float(row["sleep_hours"])
                )
            )

            history_added += 1

    conn.commit()

    # =====================================================
    # SUMMARY
    # =====================================================

    cursor.execute(
        "SELECT COUNT(*) FROM students"
    )

    total_students = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM academic_records"
    )

    total_academic = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM student_history"
    )

    total_history = cursor.fetchone()[0]

    conn.close()

    print()
    print("======================================")
    print("Migration completed successfully!")
    print("======================================")

    print(
        f"New students added: {students_added}"
    )

    print(
        f"New academic records: {academic_added}"
    )

    print(
        f"New history records: {history_added}"
    )

    print()
    print(
        f"Total students: {total_students}"
    )

    print(
        f"Total academic records: {total_academic}"
    )

    print(
        f"Total history records: {total_history}"
    )

    print("======================================")


if __name__ == "__main__":
    migrate()