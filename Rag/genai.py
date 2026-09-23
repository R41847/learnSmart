import os
import json
from pathlib import Path

from google import genai

from retriever import retrieve_resources


# =========================
# Gemini Client
# =========================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set. "
        "Please add your Gemini API key as an environment variable."
    )


client = genai.Client(
    api_key=API_KEY
)


def _parse_json_response(response_text):
    """Parse Gemini JSON output, including fenced JSON responses."""
    cleaned = response_text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        cleaned = cleaned.rsplit("```", 1)[0].strip()

    return json.loads(cleaned)


def grade_assignment(assignment_title, questions, student_answers):
    """Grade free-text assignment answers with question-level feedback."""
    question_prompt = "\n\n".join(
        f"""QUESTION {index}
Question: {item["question"]}
Model answer: {item["model_answer"]}
Student answer: {student_answers[index - 1]}
Maximum points: {item["max_points"]}"""
        for index, item in enumerate(questions, 1)
    )

    prompt = f"""
You are LearnSmart AI, an educational assignment grader.

Grade the student's answers for the assignment "{assignment_title}".
Evaluate each answer independently against its model answer.
Accept equivalent wording, valid reasoning, and correct answers that
are expressed differently from the model answer. Do not require
word-for-word matching.

Return ONLY valid JSON with this exact shape:
{{
  "questions": [
    {{
      "question_number": 1,
      "points_awarded": 0,
      "max_points": 1,
      "feedback": "Specific, concise feedback for the student."
    }}
  ],
  "overall_feedback": "A concise summary of strengths and areas to improve."
}}

Rules:
- Include exactly one result for every question, in question order.
- points_awarded must be between 0 and max_points.
- max_points must match the supplied maximum.
- Give useful written feedback for every answer.
- Do not include markdown or extra text outside the JSON.

{question_prompt}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    try:
        result = _parse_json_response(response.text)
    except (AttributeError, json.JSONDecodeError) as error:
        raise ValueError("Gemini returned invalid grading JSON.") from error

    graded_questions = result.get("questions")
    if not isinstance(graded_questions, list) or len(graded_questions) != len(questions):
        raise ValueError("Gemini returned an incomplete assignment grade.")

    normalized_questions = []
    for index, (question, graded) in enumerate(
        zip(questions, graded_questions), 1
    ):
        try:
            awarded = float(graded["points_awarded"])
            max_points = float(question["max_points"])
            feedback = str(graded["feedback"]).strip()
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Gemini returned an invalid question grade.") from error

        if not feedback or not 0 <= awarded <= max_points:
            raise ValueError("Gemini returned an out-of-range question grade.")

        normalized_questions.append(
            {
                "question_number": index,
                "points_awarded": awarded,
                "max_points": max_points,
                "feedback": feedback
            }
        )

    earned_points = sum(item["points_awarded"] for item in normalized_questions)
    max_points = sum(item["max_points"] for item in normalized_questions)

    return {
        "questions": normalized_questions,
        "overall_feedback": str(
            result.get("overall_feedback", "")
        ).strip(),
        "earned_points": earned_points,
        "max_points": max_points,
        "percentage": (earned_points / max_points * 100)
        if max_points
        else 0
    }


# =========================
# Generate Personalized Plan
# =========================

def generate_learning_plan(student_profile, language="en"):

    # Retrieve relevant resources
    resources = retrieve_resources(
        student_profile,
        top_k=5
    )

    if not resources:

        return "No suitable learning resources were found."


    # Build retrieved context
    context = ""

    suffix = "_ar" if language == "ar" else ""

    for i, resource in enumerate(resources, 1):
        metadata = resource["metadata"]
        content = (
            resource.get("content_ar")
            if language == "ar"
            else resource.get("content_en")
        )
        if not content:
            content = resource.get("content_en", "")

        context += f"""

RESOURCE {i}

Title:
{metadata.get(f"title{suffix}")}

Topic:
{metadata.get(f"topic{suffix}")}

Difficulty:
{metadata.get(f"difficulty{suffix}")}

Content Type:
{metadata.get(f"content_type{suffix}")}

Learning Style:
{metadata.get(f"learning_style{suffix}")}

Resource Information:
{content}

"""


    # =========================
    # Prompt
    # =========================

    prompt = f"""
You are LearnSmart AI, an educational assistant
that helps teachers create personalized learning
plans for students.

You MUST base your recommendations on the retrieved
learning resources provided below.

Do not invent courses or resources that are not present
in the retrieved context.

Student Profile:

Performance Level:
{student_profile.get("performance_level")}

Age:
{student_profile.get("age")}

Education Level:
{student_profile.get("education_level")}

Learning Style:
{student_profile.get("learning_style")}

Preferred Topics:
{student_profile.get("preferred_topics")}

Weak Areas:
{student_profile.get("weak_areas")}

Study Hours:
{student_profile.get("study_hours")}

Attendance:
{student_profile.get("attendance")}


Retrieved Learning Resources:
{context}


Create a concise personalized learning plan.

Your response should contain:

1. Student Learning Insight
2. Recommended Resources
3. 3-Step Learning Plan
4. Suggested Practice Activity
5. Teacher Recommendation

Keep the language simple and practical.

Only recommend resources that appear
in the retrieved context.
"""

    if language == "ar":
        prompt += """

Write the entire learning plan in clear, simple Modern Standard Arabic.
Use the Arabic resource names from the retrieved context.
"""
    else:
        prompt += """

Write the entire learning plan in clear, simple English.
"""


    # =========================
    # Gemini
    # =========================

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


# =========================
# Test
# =========================

if __name__ == "__main__":

    student = {

        "performance_level": "At Risk",

        "age": 10,

        "education_level": "Primary",

        "learning_style": "Visual",

        "preferred_topics": "Math",

        "weak_areas": "Low assignment performance",

        "study_hours": 2,

        "attendance": 70
    }


    result = generate_learning_plan(
        student
    )

    print("\n")
    print("=" * 60)
    print("LEARN SMART AI")
    print("=" * 60)
    print(result)