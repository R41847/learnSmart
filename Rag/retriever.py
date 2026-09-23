from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# =========================
# Paths
# =========================

RAG_DIR = Path(__file__).resolve().parent

CHROMA_DIR = RAG_DIR / "chroma_db"


# =========================
# Load embedding model
# =========================

model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# Connect to ChromaDB
# =========================

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_collection(
    name="learning_resources"
)


# =========================
# Retrieve resources
# =========================

def retrieve_resources(
    student_profile,
    top_k=5
):
    """
    Retrieve learning resources relevant
    to the student's profile.
    """

    query = f"""
    Find learning resources for a student with:

    Performance Level: {student_profile.get("performance_level", "Unknown")}

    Age: {student_profile.get("age", "Unknown")}

    Education Level: {student_profile.get("education_level", "Unknown")}

    Learning Style: {student_profile.get("learning_style", "Unknown")}

    Preferred Topics: {student_profile.get("preferred_topics", "Unknown")}

    Weak Areas: {student_profile.get("weak_areas", "Unknown")}

    Study Hours: {student_profile.get("study_hours", "Unknown")}

    Attendance: {student_profile.get("attendance", "Unknown")}
    """

    query_embedding = model.encode(
        [query]
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    resources = []

    if not results["documents"]:
        return resources

    for i in range(len(results["documents"][0])):

        resource = {
            "content_en": results["documents"][0][i],
            "content_ar": results["metadatas"][0][i].get("content_ar", ""),
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
            if "distances" in results
            else None
        }

        resources.append(resource)

    return resources


# =========================
# Simple test
# =========================

if __name__ == "__main__":

    test_student = {
        "performance_level": "At Risk",
        "age": 10,
        "education_level": "Primary",
        "learning_style": "Visual",
        "preferred_topics": "Math",
        "weak_areas": "Low assignment performance",
        "study_hours": 2,
        "attendance": 70
    }

    results = retrieve_resources(
        test_student,
        top_k=5
    )

    print("\nRetrieved Resources:\n")

    for i, resource in enumerate(results, 1):

        print(f"--- Resource {i} ---")

        print(
            resource["metadata"]
        )

        print(resource["content_en"])

        print()