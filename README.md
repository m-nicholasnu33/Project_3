# ScriptureSeeker: An AI-Powered Bible Verse Recommendation System

**Team CRAM**  
Team Members: Alex Baraban, Catherine Wanko, Michael Nicholas, Ryan Hough  
**Submission Date**: October 31, 2024  
**Instructor**: Kevin  

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Industry](#project-industry)
3. [Data Sources](#data-sources)
4. [Key Research and Exploration Questions](#key-research-and-exploration-questions)
5. [Licensing and Legal Terms](#licensing-and-legal-terms)
6. [Additional Information](#additional-information)
7. [References](#references)

## Project Overview

**ScriptureSeeker** aims to develop a supervised learning model that assigns embeddings to Bible verses from the King James Version. Utilizing cosine similarity, the system retrieves the most contextually relevant verses in response to user queries. The goal is to create a model that efficiently provides appropriate biblical references based on input topics or questions.

## Project Industry

Self-Help, Inspiration

## Data Sources

- **Primary Data Source**: [Kaggle Bible Dataset (King James Version)](https://www.kaggle.com/datasets/oswinrh/bible)  
  We will use the `t_kjv` file from this dataset.
- **Secondary Data Source**: *To Be Determined*

## Key Research and Exploration Questions

While this project primarily focuses on a query model using Retrieval Augmented Generation (RAG), the key challenges include:

- Developing a model that effectively retrieves relevant Bible verses based on specific input topics or queries.
- Testing the effectiveness of the query results using cosine similarity to determine the relevance between the user query and the Bible verses.

## Licensing and Legal Terms

This project is released under the [Creative Commons Zero v1.0 Universal License](https://creativecommons.org/publicdomain/zero/1.0/legalcode.en#copyright):

- **No Copyright**: Anyone is free to copy, modify, distribute, perform, and display the work, even for commercial purposes, all without asking permission.

## Additional Information

We are creating a Gradio app with the following functionality:

## References


### Gradio App Implementation

```python
import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
import re
from scipy.spatial import distance

# Initialize the language model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

def find_best_verses(user_query):
    # Get embeddings for the user query
    uqe = get_embeddings(user_query).data[0].embedding
    scores = []
    
    # Iterate over the dataset to calculate distances
    for i, row in filtered_df.iterrows():
        dist = distance.euclidean(row["query_embeddings"], uqe)
        scores.append((dist, i))
    
    # Sort by distance and select the top 5 matches
    scores = sorted(scores, key=lambda x: x[0])[:5]
    
    # Collect the best matching verses
    best_verses = [
        f"{i + 1}. {filtered_df.iloc[index]['t']}" for i, (_, index) in enumerate(scores)
    ]
    
    # Format the output as a string
    verses_str = "\n".join(best_verses)
    
    # Evaluate relevance using LangChain
    evaluation, yes_ratio = evaluate_relevance(user_query, verses_str)
    
    # Return both the verses, evaluation result, and yes ratio
    return f"Best Verses:\n{verses_str}\n\nEvaluation:\n{evaluation}\n\nYes Ratio: {yes_ratio:.2%}"

def evaluate_relevance(user_query, verses):
    # Define the evaluation prompt template
    prompt = PromptTemplate(
        input_variables=["query", "verses"],
        template=(
            "Given the query: '{query}' and the following verses:\n\n"
            "{verses}\n\n"
            "Rate each verse with either a yes or no, yes being relevant advice and no being irrelevant advice to the query. Also give a brief explanation why you chose your answer."
            "Format your answers exactly as:\n"
            "1. yes\n"
            "2. no\n"
            "3. yes\n"
            "Provide brief explanations below the answers."
        )
    )
    
    # Format the prompt with the user's query and verses
    formatted_prompt = prompt.format(query=user_query, verses=verses)
    
    # Use LangChain to generate an evaluation
    messages = [HumanMessage(content=formatted_prompt)]
    response = llm(messages).content
    
    # Extract 'yes' or 'no' answers from the response
    answers = re.findall(r"\b(yes|no)\b", response.lower())
    
    # Calculate the ratio of 'yes' answers
    yes_count = answers.count("yes")
    total_count = len(answers)
    yes_ratio = yes_count / total_count if total_count > 0 else 0
    
    # Return the response and yes ratio
    return response, yes_ratio

# Create the Gradio interface
interface = gr.Interface(
    fn=find_best_verses,  # Function to handle input
    inputs=gr.Textbox(label="What would you like to seek advice about?"),  # Input box
    outputs="text",  # Output displayed as text
    title="Advice Seeker",  # Title of the app
    description="Enter a topic you need advice on, and we will return the top 5 Bible verses to help you with your problems."  # Brief description
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch()  