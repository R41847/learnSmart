from pathlib import Path
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

CSV_FILE = BASE_DIR / "optimized_learning_resource_recommendations_200.csv"

CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"


# =========================
# Load dataset
# =========================
print("Loading dataset...")

df = pd.read_csv(CSV_FILE)

print(f"Dataset loaded: {len(df)} rows")


# =========================
# Create text documents
# =========================
documents = []

for _, row in df.iterrows():

    text = f"""
Learning Resource Information:

Course ID: {row['course_id']}
Course Title: {row['title']}
Topic: {row['topic']}
Difficulty: {row['difficulty']}
Content Type: {row['content_type']}

Learner Information:

Age: {row['age']}
Education Level: {row['education_level']}
Learning Style: {row['learning_style']}
Preferred Topics: {row['preferred_topics']}

Learning Performance:

Engagement Score: {row['engagement_score']}
Completion Status: {row['completion_status']}
Predicted Engagement Score: {row['predicted_engagement_score']}
Assessment Score: {row['assessment_score']}
Feedback Score: {row['feedback_score']}
"""

    documents.append(text.strip())


# =========================
# Embedding model
# =========================
print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# =========================
# Generate embeddings
# =========================
print("Creating embeddings...")

embeddings = model.encode(
    documents,
    show_progress_bar=True
)

print("Embeddings created.")


# =========================
# ChromaDB
# =========================
print("Creating ChromaDB...")

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

# Delete old collection if it exists
try:
    client.delete_collection("learning_resources")
    print("Old collection deleted.")
except Exception:
    pass


collection = client.create_collection(
    name="learning_resources"
)


# =========================
# Store data
# =========================
ids = [
    f"resource_{i}"
    for i in range(len(documents))
]

metadatas = []

for _, row in df.iterrows():

    metadata = {
        "course_id": str(row["course_id"]),
        "title": str(row["title"]),
        "topic": str(row["topic"]),
        "difficulty": str(row["difficulty"]),
        "content_type": str(row["content_type"]),
        "learning_style": str(row["learning_style"]),
        "preferred_topics": str(row["preferred_topics"])
    }

    metadatas.append(metadata)


collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)


print("\n================================")
print("RAG Knowledge Base Ready!")
print("================================")
print(f"Documents stored: {len(documents)}")
print(f"Database location: {CHROMA_DIR}")