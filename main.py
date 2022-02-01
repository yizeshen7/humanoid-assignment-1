''' Filename:   tfidf.py
    Author:     Aditti Ramsisaria
    Brief:      This file is used to find cosine similarity of preprocessed 
                news articles weighted using tfidf
'''

'''
Document 1:
the sky is blue. 
Terms: 4
6x1 vector
[the blue sky is today pretty]
[-, -, -, -, 0, 0]

Dcoment 2:
the sky is pretty today.
5 terms
'''

import pandas as pd
from collections import Counter
from flask import Flask, request
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
df = pd.read_csv("tagged_data.csv")

# using tfidf vectoriser and cosine similarity
tvec = TfidfVectorizer()
# sparse matrix of vectors or descriptions
tfidf_matrix = tvec.fit_transform(df.preprocessed)

# 2 vectors a, b
# a.b = |a||b|cos(theta)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
article_indices = pd.Series(df.index, index = df["title"]).drop_duplicates()

# get titles of top 10 most similar articles to input title
@app.route('/recommend', methods=["GET", "POST"])
def get_recommendations():
    if request.method == "POST":
        start_time = time.time()
        input_title = request.form.get("title")
        if not input_title:
            return 'Please input article title'

        # get index of article that matches the title
        index = article_indices[input_title]
        # get pairwise similarity scores of that article
        similarity_scores = list(enumerate(cosine_sim[index]))
        # sort articles based on similarity scores
        similarity_scores = sorted(similarity_scores, key = lambda x:x[1], reverse = True)

        # get scores for 10 most similar articles
        top_similarity = similarity_scores[1:11]
        top_indices = [score[0] for score in top_similarity]
        top_scores = [score[1] for score in top_similarity]
        top_titles = df["title"].iloc[top_indices]

        # get recommended category
        top_categories = df["category"].iloc[top_indices]
        category_count = Counter(top_categories.tolist())
        return {
            "recommended titles" : top_titles.tolist(),
            "scores" : top_scores,
            "recommended category" : category_count.most_common(1)[0][0],
            "message" : "results found successfully",
            "plt" : time.time() - start_time
        }
    return "This is not a POST!" 

if __name__ == "__main__":
    app.run(debug=True, port=5000)
