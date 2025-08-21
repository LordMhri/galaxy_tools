import streamlit as st
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, util

# --- CONFIGURATION ---
MODEL_PATH = 'fine_tuned_galaxy_model'
EMBEDDINGS_PATH = 'tool_embeddings.npy'
IDS_PATH = 'tool_ids.pkl'
# Optional: Load the original data to show more info
DATA_PATH = 'cleaned_data.csv'

# --- RESOURCE LOADING ---
# Use Streamlit's caching to load the model and data only once.
@st.cache_resource
def load_model():
    """Loads the fine-tuned SentenceTransformer model."""
    return SentenceTransformer(MODEL_PATH)

@st.cache_data
def load_data():
    """Loads the embeddings and corresponding IDs."""
    with open(IDS_PATH, 'rb') as f:
        ids = pickle.load(f)
    embeddings = np.load(EMBEDDINGS_PATH)
    return ids, embeddings

# --- SEARCH FUNCTIONALITY ---
def perform_search(query, model, tool_ids, tool_embeddings, top_k=5):
    """
    Performs semantic search.
    
    Returns a list of tuples: (tool_id, score)
    """
    if not query.strip():
        return []

    # 1. Encode the user's query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # 2. Calculate cosine similarity between the query and all tool embeddings
    # util.cos_sim is highly efficient
    cosine_scores = util.cos_sim(query_embedding, tool_embeddings)

    # 3. Get the top_k results
    # torch.topk returns the values and their indices
    top_results = torch.topk(cosine_scores, k=min(top_k, len(tool_embeddings)), dim=1)

    # 4. Format the results
    results = []
    for score, idx in zip(top_results[0][0], top_results[1][0]):
        tool_id = tool_ids[idx.item()]
        results.append((tool_id, score.item()))
        
    return results

# --- STREAMLIT UI ---
st.title("Galaxy Tool Semantic Search 🌌")
st.write("Enter a description of what you want to do, and the search will find the most relevant tools.")

# Load resources
model = load_model()
tool_ids, tool_embeddings = load_data()

# User input
user_query = st.text_input("Search for a tool:", "")

# Search button
if st.button("Search"):
    if user_query:
        st.write("---")
        st.subheader("Search Results:")
        
        # Perform search and display results
        search_results = perform_search(user_query, model, tool_ids, tool_embeddings, top_k=10)
        
        if not search_results:
            st.warning("No results found. Try a different query.")
        else:
            for i, (tool_id, score) in enumerate(search_results):
                st.info(f"**{i+1}. Tool ID:** `{tool_id}`\n\n**Similarity Score:** `{score:.4f}`")
    else:
        st.warning("Please enter a search query.")