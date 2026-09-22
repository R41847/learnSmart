import pandas as pd


DATA_FILE = "activity_clean.csv"


def load_data():
    return pd.read_csv(DATA_FILE)


def analyze_topic_performance(df):

    topic_summary = (
        df.groupby("topic")
        .agg(
            students=("topic", "count"),
            avg_activity=("activity_score", "mean"),
            avg_absence=("absence_score", "mean"),
            high_performance=("performance_level", lambda x: (x == "High").mean()),
            medium_performance=("performance_level", lambda x: (x == "Medium").mean()),
            low_performance=("performance_level", lambda x: (x == "Low").mean())
        )
        .reset_index()
    )

    return topic_summary


def calculate_topic_risk(topic_summary):

    # Higher activity is generally a positive behavioral signal.
    # Higher absence is a negative signal.
    # Higher low-performance ratio is a negative signal.

    topic_summary["risk_score"] = (
        (1 - topic_summary["high_performance"]) * 0.4
        + topic_summary["avg_absence"] * 0.3
        + (1 - topic_summary["avg_activity"] / 100) * 0.3
    )

    topic_summary["risk_level"] = pd.cut(
        topic_summary["risk_score"],
        bins=[-1, 0.35, 0.60, 1.0],
        labels=["Low", "Medium", "High"]
    )

    return topic_summary.sort_values(
        "risk_score",
        ascending=False
    )


def show_results(topic_summary):

    print("\n========== TOPIC ANALYSIS ==========")

    print(
        topic_summary[
            [
                "topic",
                "students",
                "avg_activity",
                "avg_absence",
                "high_performance",
                "low_performance",
                "risk_score",
                "risk_level"
            ]
        ].round(3)
    )


if __name__ == "__main__":

    df = load_data()

    topic_summary = analyze_topic_performance(df)

    topic_summary = calculate_topic_risk(topic_summary)

    show_results(topic_summary)