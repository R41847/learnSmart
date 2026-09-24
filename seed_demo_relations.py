"""Seed a small, connected teacher -> student -> parent demo relationship.

This is intentionally separate from migrate_data.py and is safe to run more
than once against the existing LearnSmart SQLite database.
"""

import sqlite3

from database import DATABASE_NAME, create_tables


TEACHER_CODE = "Ms. Sara Hassan"
SCHOOL_NAME = "Nile Future School"
CLASS_NAME = "5A"
STUDENT_NAMES = (
    "Ahmed Ali",
    "Omar Mohamed",
    "Youssef Hassan",
    "Jana Ahmed",
)
PARENT_EMAIL = "ahmed123@gmail.com"


def seed_demo_relations():
    create_tables()
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row

    try:
        conn.execute("BEGIN")

        teacher = conn.execute(
            "SELECT id FROM teachers WHERE teacher_code = ?",
            (TEACHER_CODE,),
        ).fetchone()
        if teacher is None:
            cursor = conn.execute(
                """
                INSERT INTO teachers (teacher_code, subject)
                VALUES (?, ?)
                """,
                (TEACHER_CODE, "General Studies"),
            )
            teacher_id = cursor.lastrowid
        else:
            teacher_id = teacher["id"]

        school = conn.execute(
            "SELECT id FROM schools WHERE name = ?",
            (SCHOOL_NAME,),
        ).fetchone()
        if school is None:
            cursor = conn.execute(
                "INSERT INTO schools (name) VALUES (?)",
                (SCHOOL_NAME,),
            )
            school_id = cursor.lastrowid
        else:
            school_id = school["id"]

        conn.execute(
            "UPDATE teachers SET school_id = ? WHERE id = ?",
            (school_id, teacher_id),
        )

        students = conn.execute(
            """
            SELECT id
            FROM students
            ORDER BY id
            LIMIT ?
            """,
            (len(STUDENT_NAMES),),
        ).fetchall()
        if len(students) != len(STUDENT_NAMES):
            raise RuntimeError(
                f"Expected at least {len(STUDENT_NAMES)} migrated students, "
                f"found {len(students)}."
            )

        for student, student_name in zip(students, STUDENT_NAMES):
            conn.execute(
                """
                UPDATE students
                SET student_name = ?,
                    teacher_name = ?,
                    teacher_id = ?,
                    class_name = ?,
                    school_name = ?,
                    school_id = ?
                WHERE id = ?
                """,
                (
                    student_name,
                    TEACHER_CODE,
                    teacher_id,
                    CLASS_NAME,
                    SCHOOL_NAME,
                    school_id,
                    student["id"],
                ),
            )

        parent = conn.execute(
            """
            SELECT p.id, u.name
            FROM parents p
            JOIN users u ON u.id = p.user_id
            WHERE u.email = ? AND u.role = 'parent'
            """,
            (PARENT_EMAIL,),
        ).fetchone()
        if parent is None:
            raise RuntimeError(
                f"No existing parent account found for {PARENT_EMAIL}."
            )

        conn.execute(
            """
            INSERT OR IGNORE INTO parent_student (parent_id, student_id)
            VALUES (?, ?)
            """,
            (parent["id"], students[0]["id"]),
        )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print("Demo relational seed complete:")
    print(f"  Teacher: {TEACHER_CODE} (teachers.id={teacher_id})")
    print(f"  School: {SCHOOL_NAME}")
    print("  Students:")
    for student_name in STUDENT_NAMES:
        print(f"    - {student_name}")
    print(
        f"  Parent: {parent['name']} ({PARENT_EMAIL}) "
        f"linked to {STUDENT_NAMES[0]}"
    )


if __name__ == "__main__":
    seed_demo_relations()
