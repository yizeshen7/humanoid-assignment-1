# humanoid-assignment-1
Authors: Aditti Ramsisaria, Yize Shen

This project is a News Recommendation Engine for the BBC News Dataset based on basic NLP techniques like text embeddings.

The BBC dataset (available: http://mlg.ucd.ie/datasets/bbc.html) is first converted into a .csv file by running the create_csv.py file.

Pipeline:
The data in this file is then preprocessed and cleaned by running the data_preprocessing.py file, which saves the cleaned/tagged data as tagged_data.csv
The main.py file creates text embeddings using TFIDF for the cleaned data, and computes pairwise cosine similarities on the sparse matrix produced by tfidf.
It also recieves HTTP POST requests (localhost:5000/recommend) and finds the 10 most similar articles to the input title and returns the results as a json.
