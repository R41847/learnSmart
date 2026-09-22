import pandas as pd

INPUT_FILE = "xAPI-Edu-Data.csv"
OUTPUT_FILE = "activity_clean.csv"


def load_data():
    df = pd.read_csv(INPUT_FILE)
    print(f"Original shape: {df.shape}")
    return df


def clean_data(df):

    # Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Fix specific names
    df = df.rename(columns={
        "nationality": "nationality",
        "placeofbirth": "place_of_birth",
        "stageid": "stage",
        "gradeid": "grade_level",
        "sectionid": "section",
        "vis itedresources": "visited_resources",
        "visitedresources": "visited_resources",
        "announcementsview": "announcements_view",
        "parentansweringsurvey": "parent_answering_survey",
        "parentschoolsatisfaction": "parent_school_satisfaction",
        "studentabsencedays": "student_absence_days"
    })

    # Standardize categorical values
    df["gender"] = df["gender"].str.upper()

    df["topic"] = df["topic"].str.strip().str.title()

    # Convert activity columns to numeric
    numeric_columns = [
        "raisedhands",
        "visited_resources",
        "announcements_view",
        "discussion"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Create normalized activity score
    activity_columns = [
        "raisedhands",
        "visited_resources",
        "announcements_view",
        "discussion"
    ]

    df["activity_score"] = (
        df[activity_columns].mean(axis=1)
    )

    # Create absence score
    df["absence_score"] = df["student_absence_days"].map({
        "Under-7": 0,
        "Above-7": 1
    })

    # Convert Class to readable performance level
    df["performance_level"] = df["class"].map({
        "L": "Low",
        "M": "Medium",
        "H": "High"
    })

    return df


def save_data(df):
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved cleaned dataset to: {OUTPUT_FILE}")


def show_summary(df):

    print("\n========== CLEANED SHAPE ==========")
    print(df.shape)

    print("\n========== COLUMNS ==========")
    print(df.columns.tolist())

    print("\n========== TOPICS ==========")
    print(df["topic"].value_counts())

    print("\n========== PERFORMANCE ==========")
    print(df["performance_level"].value_counts())

    print("\n========== AVERAGE ACTIVITY BY TOPIC ==========")

    topic_activity = (
        df.groupby("topic")["activity_score"]
        .mean()
        .sort_values(ascending=False)
    )

    print(topic_activity)

    print("\n========== ABSENCE BY PERFORMANCE ==========")

    print(
        pd.crosstab(
            df["performance_level"],
            df["student_absence_days"],
            normalize="index"
        ).round(2)
    )


if __name__ == "__main__":

    df = load_data()

    df = clean_data(df)

    show_summary(df)

    save_data(df)

    print("\n========== DONE ==========")