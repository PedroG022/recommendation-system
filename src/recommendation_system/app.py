import ast

import pandas as pd
import streamlit as st
from rich import print
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

# --- Configuration ---
DATA_PATH = "./data/movies_metadata.csv"
MAX_SELECTED_MOVIES = 5
PLACEHOLDER_IMAGE_URL = "https://placehold.co/400x600?text="
CARD_IMAGE_WIDTH = 150


# --- Data Loading and Feature Engineering (Cached) ---
@st.cache_data
def load_and_prepare_data(data_path):
    try:
        movies_df = pd.read_csv(data_path, low_memory=False)
    except FileNotFoundError:
        st.error(
            f"Error: Movie metadata file not found at {data_path}. Please ensure it's there."
        )
        st.stop()

    # --- Basic Cleaning ---
    movies_df["id"] = pd.to_numeric(movies_df["id"], errors="coerce")
    movies_df = movies_df.dropna(subset=["id"])
    movies_df["id"] = movies_df["id"].astype(int)
    movies_df = movies_df.drop_duplicates(subset=["id"], keep="first")

    movies_df = movies_df[
        ["id", "title", "genres", "runtime", "original_language", "overview"]
    ]  # Keep necessary columns
    movies_df = movies_df.dropna(subset=["title"])  # Drop movies without titles
    movies_df["title"] = movies_df["title"].astype(str)  # Ensure title is string

    # --- Feature: Genres ---
    def parse_genres(genres_str):
        try:
            genres_list = ast.literal_eval(genres_str)
            return [
                d["name"] for d in genres_list if isinstance(d, dict) and "name" in d
            ]
        except Exception:
            return []

    movies_df["genre_names"] = movies_df["genres"].apply(parse_genres)
    mlb = MultiLabelBinarizer()

    movies_df_indexed = movies_df.set_index("id", drop=False)
    genre_dummies = pd.DataFrame(
        mlb.fit_transform(movies_df_indexed["genre_names"]),
        columns=mlb.classes_,
        index=movies_df_indexed.index,
    )

    # --- Feature: Runtime ---
    movies_df["runtime"] = pd.to_numeric(movies_df["runtime"], errors="coerce")
    # Use median which is often more robust to outliers than mean for runtime
    movies_df["runtime"] = movies_df["runtime"].fillna(movies_df["runtime"].median())
    # Handle edge case if median is also NaN
    if pd.isna(movies_df["runtime"].median()):
        movies_df["runtime"] = movies_df["runtime"].fillna(0)
    runtime_df = movies_df.set_index("id")["runtime"]

    # --- Feature: Language ---
    # Use value_counts on the original DataFrame before indexing if easier
    top_langs = movies_df["original_language"].value_counts().nlargest(15).index
    movies_df["lang_processed"] = movies_df["original_language"].apply(
        lambda x: x if x in top_langs else "other"
    )
    # Re-index before get_dummies
    lang_dummies = pd.get_dummies(
        movies_df.set_index("id")["lang_processed"], prefix="lang"
    )

    # --- Combine Features ---
    # Ensure indices align using the indexed DataFrame
    feature_cols_df = pd.DataFrame(
        index=movies_df_indexed.index
    )  # Start with correct index
    feature_cols_df = feature_cols_df.join(runtime_df)
    feature_cols_df = feature_cols_df.join(genre_dummies)  # Joins based on index
    feature_cols_df = feature_cols_df.join(lang_dummies)
    feature_cols_df = feature_cols_df.fillna(0)  # Fill NaNs possibly created by joins

    # --- Scaling ---
    scaler = MinMaxScaler()
    movie_ids_index = feature_cols_df.index
    movie_features_scaled = scaler.fit_transform(feature_cols_df)
    movie_features_scaled_df = pd.DataFrame(
        movie_features_scaled, columns=feature_cols_df.columns, index=movie_ids_index
    )

    # --- Mappings ---
    # Filter the original df (with titles) to only include IDs that survived processing
    valid_movie_ids = movie_features_scaled_df.index
    filtered_movies_df = (
        movies_df[movies_df["id"].isin(valid_movie_ids)]
        .drop_duplicates(subset=["id"])
        .set_index("id")
    )
    movie_id_to_title = filtered_movies_df["title"].to_dict()

    # Create title_to_movie_id carefully, handling potential duplicate titles (use first ID found)
    title_to_movie_id = {}
    for movie_id, title in movie_id_to_title.items():
        if pd.notna(title) and title not in title_to_movie_id:
            title_to_movie_id[title] = movie_id

    print(f"Data loaded. Features shape: {movie_features_scaled_df.shape}")

    return (
        movie_features_scaled_df,
        movie_id_to_title,
        title_to_movie_id,
        filtered_movies_df[["title", "overview", "genre_names"]],
    )


# --- Recommendation Function --- (Same as before)
def recommend_for_new_user(
    liked_movie_ids, n, all_movie_features_scaled, id_to_title_map
):
    valid_liked_ids = [
        mid for mid in liked_movie_ids if mid in all_movie_features_scaled.index
    ]
    if not valid_liked_ids:
        return pd.DataFrame(
            columns=["movieId", "title", "similarity_score"]
        )  # Empty if no valid IDs

    liked_features = all_movie_features_scaled.loc[valid_liked_ids]
    user_profile = liked_features.mean(axis=0).values.reshape(1, -1)
    similarities = cosine_similarity(user_profile, all_movie_features_scaled.values)
    sim_scores = pd.Series(
        similarities.flatten(), index=all_movie_features_scaled.index
    )
    sim_scores = sim_scores.drop(valid_liked_ids, errors="ignore")
    top_n_recommendations = sim_scores.sort_values(ascending=False).head(n)

    recommendations_df = pd.DataFrame(
        {
            "movieId": top_n_recommendations.index,
            "similarity_score": top_n_recommendations.values,
        }
    )
    recommendations_df["title"] = recommendations_df["movieId"].map(id_to_title_map)
    return recommendations_df[["movieId", "title", "similarity_score"]]


# --- Streamlit App ---

# --- Load Data ---
# Pass the path to the function
movie_features_scaled_df, movie_id_to_title, title_to_movie_id, movies_display_df = (
    load_and_prepare_data(DATA_PATH)
)

# Check if data loading failed
if movie_features_scaled_df is None:
    st.warning(
        "Could not load movie data. Please check the file path and data integrity."
    )
    st.stop()

# --- Initialize Session State ---
if "selected_movies" not in st.session_state:
    st.session_state.selected_movies = []  # List to store tuples of (id, title)

# --- App Title ---
st.title("🎬 My Movies")
st.subheader("Movie Recommendations")
st.write(f"Select up to {MAX_SELECTED_MOVIES} movies you like to get recommendations.")

# --- Movie Selection Area ---
st.header("1. Select Liked Movies")


# --- Callback function to handle movie addition and clearing ---
def handle_movie_selection():
    # Get the selected movie title from the widget's state using its key
    selected_title = st.session_state.movie_search_key

    if selected_title and selected_title != "":  # Check if a movie title is selected
        movie_id = title_to_movie_id.get(selected_title)

        if movie_id:
            # Check if already selected
            if any(movie[0] == movie_id for movie in st.session_state.selected_movies):
                st.warning(f"'{selected_title}' is already selected.")
            # Check if max limit reached
            elif len(st.session_state.selected_movies) >= MAX_SELECTED_MOVIES:
                st.warning(f"You have already selected {MAX_SELECTED_MOVIES} movies.")
            else:
                # Add movie to session state
                st.session_state.selected_movies.append((movie_id, selected_title))
                st.success(f"Added '{selected_title}'")
                st.session_state.movie_search_key = (
                    ""  # Set it back to the default empty string
                )
        else:
            st.error(
                f"Could not find ID for title '{selected_title}'. This shouldn't normally happen."
            )
            st.session_state.movie_search_key = ""  # Also clear if error


# --- Search Bar ---
movie_titles_list = [""] + sorted(
    [title for title in title_to_movie_id.keys() if title]
)  # Ensure titles are strings and not None/NaN, add empty first option

st.selectbox(
    "Search and select a movie:",
    options=movie_titles_list,
    index=0,  # Default to the empty string option
    key="movie_search_key",  # Assign a key to access its state
    on_change=handle_movie_selection,  # Register the callback
)

# --- Display Selected Movies ---
st.subheader("Your Selections:")

if not st.session_state.selected_movies:
    st.info("No movies selected yet. Use the search bar above.")
else:
    num_selected = len(st.session_state.selected_movies)

    # Adjust grid width dynamically, max 5 columns
    num_cols = min(num_selected, 5)
    cols = st.columns(num_cols)
    movies_to_remove = []  # Keep track of movies marked for removal

    for i, (movie_id, movie_title) in enumerate(st.session_state.selected_movies):
        col_index = i % num_cols

        with cols[col_index]:
            # Use a container for better grouping of image and button
            with st.container(border=True):
                # Apply fixed width to the image
                st.image(
                    PLACEHOLDER_IMAGE_URL
                    + movie_title.replace(" ", "+")[:20]
                    + "\\n"
                    + movie_title[20:40],
                    caption=movie_title[:30]
                    + (
                        "..." if len(movie_title) > 30 else ""
                    ),  # Truncate long captions
                    width=CARD_IMAGE_WIDTH,  # Apply fixed width
                    use_container_width="never",  # Ensure width parameter is respected
                )

                # Remove button - use unique key
                if st.button("Remove", key=f"remove_{movie_id}"):
                    # Don't remove directly here, mark for removal
                    movies_to_remove.append((movie_id, movie_title))

    # Process removals outside the loop
    if movies_to_remove:
        for mid, mtitle in movies_to_remove:
            # Find the specific tuple to remove
            movie_tuple_to_remove = next(
                (item for item in st.session_state.selected_movies if item[0] == mid),
                None,
            )

            if movie_tuple_to_remove:
                st.session_state.selected_movies.remove(movie_tuple_to_remove)
                st.toast(f"Removed '{mtitle}'")

        # Rerun to update the display AFTER processing removals
        st.rerun()

# --- Recommendation Trigger ---
st.header("2. Get Recommendations")

num_selected = len(
    st.session_state.selected_movies
)  # Recalculate after potential removals

if num_selected < 1:
    st.warning("Please select at least one movie first.")
elif num_selected < MAX_SELECTED_MOVIES:
    st.info(
        f"Select {MAX_SELECTED_MOVIES - num_selected} more movie(s) for potentially better recommendations (or click below)."
    )

# Enable button only if at least one movie is selected
recommend_button_pressed = st.button(
    "✨ Get Recommendations", disabled=(num_selected == 0), type="primary"
)

# --- Display Recommendations ---
if recommend_button_pressed and num_selected > 0:
    liked_ids = [movie[0] for movie in st.session_state.selected_movies]

    with st.spinner("Finding movies you might like..."):
        recommendations = recommend_for_new_user(
            liked_movie_ids=liked_ids,
            n=15,  # Number of recommendations to generate
            all_movie_features_scaled=movie_features_scaled_df,
            id_to_title_map=movie_id_to_title,
        )

    st.subheader("Top Recommendations For You:")

    if not recommendations.empty and movies_display_df is not None:
        # Merge details for display using the correct index
        recommendations_display = recommendations.merge(
            movies_display_df[
                ["title", "overview", "genre_names"]
            ],  # Select columns from the re-indexed df
            left_on="movieId",
            right_index=True,  # Merge based on the index of movies_display_df
            how="left",
        )

        # Handle potential missing genres after merge
        recommendations_display["genres"] = recommendations_display[
            "genre_names"
        ].apply(lambda x: ", ".join(x) if isinstance(x, list) and x else "N/A")

        st.dataframe(
            recommendations_display[
                ["title_x", "similarity_score", "genres", "overview"]
            ],
            column_config={
                "title_x": "Title",
                "similarity_score": st.column_config.ProgressColumn(
                    "Match Score",
                    help="How similar the movie is to your selections (0-1)",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                ),
                "genres": st.column_config.TextColumn("Genres", width="medium"),
                "overview": st.column_config.TextColumn("Overview", width="large"),
            },
            hide_index=True,
            use_container_width=True,
        )

    elif recommendations.empty:
        st.warning("Could not generate recommendations based on the selected movies.")
    else:
        st.error("There was an issue displaying recommendation details.")
