import ast
import numpy as np
import pandas as pd
import streamlit as st
from openai.embeddings_utils import cosine_similarity
import openai

# Set your OpenAI API key
openai.api_key = st.secrets["mykey"] 

# Load Data & Embeddings
try:
    df = pd.read_csv("qa_dataset_with_embeddings.csv")
    # Convert string embeddings to numpy arrays
    df['Question_Embedding'] = df['Question_Embedding'].apply(ast.literal_eval)
except FileNotFoundError:
    st.error("qa_dataset_with_embeddings.csv not found. Please upload the file.")
    st.stop() # Stop execution if file not found


# Embedding Model (using OpenAI's text-embedding-ada-002)
def get_embedding(text, model="text-embedding-ada-002"):
    try:
        embedding = openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
        return embedding
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

# Question Answering Logic
def find_best_answer(user_question, df):
    user_question_embedding = get_embedding(user_question)

    if user_question_embedding is None: # Handle embedding generation errors
        return "Error processing your question. Please try again."

    df['Similarity'] = df['Question_Embedding'].apply(lambda x: cosine_similarity(x, user_question_embedding))

    most_similar_index = df['Similarity'].idxmax()
    max_similarity = df['Similarity'].max()

    similarity_threshold = 0.6  # Adjust as needed
    if max_similarity >= similarity_threshold:
        best_answer = df.loc[most_similar_index, 'Answer']
        return best_answer
    else:
        return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?"


# Streamlit Interface
st.title("Smart FAQ Assistant (Heart, Lung, Blood Health)")

user_question = st.text_input("Enter your question about heart, lung, or blood health:")
search_button = st.button("Find Answer")

if search_button:
    if not user_question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching for the best answer..."):  # Display a spinner while searching
            answer = find_best_answer(user_question, df)
            st.write("## Answer:")
            st.write(answer)


# File uploader (Optional - if you want users to upload the CSV)
uploaded_file = st.file_uploader("Upload your CSV file (qa_dataset_with_embeddings.csv)", type="csv")
if uploaded_file is not None:
    try:
      df = pd.read_csv(uploaded_file)
      df['Question_Embedding'] = df['Question_Embedding'].apply(ast.literal_eval).apply(np.array)
      st.success("File uploaded and processed successfully!")
    except Exception as e:
      st.error(f"Error processing uploaded file: {e}")
