from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np

# Load the BGE model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Updated abstract (lung cancer study)
abstract = """
In recent times, lung cancer is the most common cause of mortality in both men and women around the world. 
Lung cancer is the second most well-known disease after heart disease. Although lung cancer prevention is impossible, 
early detection of lung cancer can effectively treat it at an early stage, significantly increasing a patient’s survival rate. 
To detect and diagnose lung cancer in its early stages, a variety of data analysis and machine learning techniques have been applied. 
In this paper, we applied supervised machine learning algorithms like SVM (Support Vector Machine), ANN (Artificial Neural Networks), 
MLR (Multiple Linear Regression), and RF (Random Forest) to detect early stages of lung tumors. 
The main purpose of this study is to examine the success of machine learning algorithms in detecting lung cancer at an early stage. 
When compared to all other supervised machine learning algorithms, the Random Forest model produces the highest result, 
with a 99.99% accuracy rate.
"""

# Convert abstract into embedding once
abstract_embedding = model.encode([abstract], convert_to_numpy=True)

# Interactive loop for questions
while True:
    question = input("\nEnter your question (or 'exit' to quit): ")
    if question.lower() == "exit":
        break

    # Convert question into embedding
    question_embedding = model.encode([question], convert_to_numpy=True)

    # Cosine similarity
    cosine_sim = cosine_similarity(abstract_embedding, question_embedding)[0][0]

    # Euclidean distance
    euclidean_dist = euclidean_distances(abstract_embedding, question_embedding)[0][0]

    print(f"Cosine Similarity: {cosine_sim:.4f}")
    print(f"Euclidean Distance: {euclidean_dist:.4f}")
