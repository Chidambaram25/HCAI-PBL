import pandas as pd

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

INPUT_PATH = "data/movie_metadata.csv"

df = pd.read_csv(INPUT_PATH)

print("Original number of rows:", len(df))


# --------------------------------------------------
# 2. Remove exact duplicate rows
# --------------------------------------------------

before = len(df)

df = df.drop_duplicates()

after = len(df)

print("Exact duplicate rows removed:", before - after)
print("Rows after removing duplicates:", after)


# --------------------------------------------------
# 3. Clean movie titles
# --------------------------------------------------

df["movie_title"] = df["movie_title"].str.strip()


# --------------------------------------------------
# 4. Handle missing numerical values
# --------------------------------------------------

df["duration"] = df["duration"].fillna(
    df["duration"].median()
)

df["title_year"] = df["title_year"].fillna(
    df["title_year"].median()
)


# --------------------------------------------------
# 5. Handle missing categorical values
# --------------------------------------------------

categorical_columns = [
    "language",
    "country",
    "content_rating"
]

for column in categorical_columns:
    df[column] = df[column].fillna("Unknown")


# --------------------------------------------------
# 6. Check missing values
# --------------------------------------------------

selected_features = [
    "genres",
    "duration",
    "title_year",
    "language",
    "country",
    "content_rating"
]

print("\nMissing values after preprocessing:")

print(df[selected_features].isnull().sum())


# --------------------------------------------------
# 7. Show some processed movies
# --------------------------------------------------

print("\nFirst 10 processed movies:")

print(
    df[
        [
            "movie_title",
            "genres",
            "duration",
            "title_year",
            "language",
            "country",
            "content_rating"
        ]
    ].head(10)
)