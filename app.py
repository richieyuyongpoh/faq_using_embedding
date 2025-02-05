import ast
import pandas as pd
import streamlit as st
from openai.embeddings_utils import cosine_similarity
import openai

# Set your OpenAI API key
openai.api_key = st.secrets["mykey"] 

df = pd.read_csv("qa_dataset_with_embeddings.csv")

# Convert the string embeddings back to lists
df['Question_Embedding'] = df['Question_Embedding'].apply(ast.literal_eval)

def get_embedding(text, model="text-embedding-ada-002"):
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def find_best_answer(user_question):
   # Get embedding for the user's question
   user_question_embedding = get_embedding(user_question)

   # Calculate cosine similarities for all questions in the dataset
   df['Similarity'] = df['Question_Embedding'].apply(lambda x: cosine_similarity(x, user_question_embedding))

   # Find the most similar question and get its corresponding answer
   most_similar_index = df['Similarity'].idxmax()
   max_similarity = df['Similarity'].max()

   # Set a similarity threshold to determine if a question is relevant enough
   similarity_threshold = 0.75  # You can adjust this value

   if max_similarity >= similarity_threshold:
      best_answer = df.loc[most_similar_index, 'Answer']
      return best_answer
   else:
      return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?"


# Streamlit Interface
st.title("Smart FAQ Assistant (Heart, Lung, Blood Health)")

user_question = st.text_input("Ask a question","Who will have Cardiomyopathy?")
search_button = st.button("Find Answer")

if search_button:
    if not user_question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching for the best answer..."):  # Display a spinner while searching
            answer = find_best_answer(user_question)
            st.write("## Answer:")
            st.write(answer)

