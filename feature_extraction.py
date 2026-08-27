import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Load and preprocess dataset
# --------------------------------------------------

INPUT_PATH = "data/movie_metadata.csv"

df = pd.read_csv(INPUT_PATH)

# Remove exact duplicate rows
df = df.drop_duplicates()

# Clean movie titles
df["movie_title"] = df["movie_title"].str.strip()

# Handle missing numerical values
df["duration"] = df["duration"].fillna(df["duration"].median())
df["title_year"] = df["title_year"].fillna(df["title_year"].median())

# Handle missing categorical values
categorical_columns = [
    "language",
    "country",
    "content_rating"
]

for column in categorical_columns:
    df[column] = df[column].fillna("Unknown")


# --------------------------------------------------
# 2. Prepare multi-label columns
# --------------------------------------------------

def split_values(value):
    return [item.strip() for item in str(value).split("|")]


df["genres_list"] = df["genres"].apply(split_values)
df["country_list"] = df["country"].apply(split_values)


# --------------------------------------------------
# 3. Encode genres
# --------------------------------------------------

genre_encoder = MultiLabelBinarizer()

genre_features = genre_encoder.fit_transform(
    df["genres_list"]
)

genre_columns = [
    f"genre_{genre}"
    for genre in genre_encoder.classes_
]

genre_df = pd.DataFrame(
    genre_features,
    columns=genre_columns,
    index=df.index
)


# --------------------------------------------------
# 4. Encode countries
# --------------------------------------------------

country_encoder = MultiLabelBinarizer()

country_features = country_encoder.fit_transform(
    df["country_list"]
)

country_columns = [
    f"country_{country}"
    for country in country_encoder.classes_
]

country_df = pd.DataFrame(
    country_features,
    columns=country_columns,
    index=df.index
)


# --------------------------------------------------
# 5. Encode language and content rating
# --------------------------------------------------

one_hot_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

other_features = one_hot_encoder.fit_transform(
    df[["language", "content_rating"]]
)

other_columns = one_hot_encoder.get_feature_names_out(
    ["language", "content_rating"]
)

other_df = pd.DataFrame(
    other_features,
    columns=other_columns,
    index=df.index
)


# --------------------------------------------------
# 6. Standardize numerical features
# --------------------------------------------------

scaler = StandardScaler()

numerical_features = scaler.fit_transform(
    df[["duration", "title_year"]]
)

numerical_df = pd.DataFrame(
    numerical_features,
    columns=["duration_scaled", "year_scaled"],
    index=df.index
)


# --------------------------------------------------
# 7. Combine all features
# --------------------------------------------------

feature_df = pd.concat(
    [
        genre_df,
        country_df,
        other_df,
        numerical_df
    ],
    axis=1
)


# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print("Number of movies:", len(feature_df))
print("Number of features:", len(feature_df.columns))

print("\nFeature columns:")
for column in feature_df.columns:
    print("-", column)


# --------------------------------------------------
# 9. Display first movie vector
# --------------------------------------------------

print("\nFirst movie:")
print(df.iloc[0]["movie_title"])

print("\nFeature vector:")
print(feature_df.iloc[0].to_string())


# --------------------------------------------------
# 10. Check feature matrix shape
# --------------------------------------------------

print("\nFeature matrix shape:")
print(feature_df.shape)

# --------------------------------------------------
# 11. Save processed movie data
# --------------------------------------------------

processed_movies = pd.concat(
    [
        df[["movie_title", "genres", "duration", "title_year",
            "language", "country", "content_rating"]].reset_index(drop=True),
        feature_df.reset_index(drop=True)
    ],
    axis=1
)

OUTPUT_PATH = "data/processed_movies.csv"

processed_movies.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nProcessed dataset saved to:")
print(OUTPUT_PATH)

print("\nProcessed dataset shape:")
print(processed_movies.shape)