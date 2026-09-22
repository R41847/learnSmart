import os
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


# =========================
# Generate Personalized Plan
# =========================

def generate_learning_plan(student_profile):

    # Retrieve relevant resources
    resources = retrieve_resources(
        student_profile,
        top_k=5
    )

    if not resources:

        return "No suitable learning resources were found."


    # Build retrieved context
    context = ""

    for i, resource in enumerate(resources, 1):

        context += f"""

RESOURCE {i}

Title:
{resource["metadata"].get("title")}

Topic:
{resource["metadata"].get("topic")}

Difficulty:
{resource["metadata"].get("difficulty")}

Content Type:
{resource["metadata"].get("content_type")}

Learning Style:
{resource["metadata"].get("learning_style")}

Resource Information:
{resource["content"]}

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