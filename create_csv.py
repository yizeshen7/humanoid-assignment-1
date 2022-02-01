''' Filename:   create_csv.py
    Author:     Aditti Ramsisaria
    Brief:      This file is used to create a csv file for the bbc 
                news full text dataset
'''
import pandas as pd
import numpy as np
import os

data_folder = "./bbc/"
categories = ["business", "entertainment", "politics", "sport", "tech"]

article_descriptions = []
article_categories = []
article_titles = []

# traverse each file in ./bbc
for folder in categories:
    folder_path = "./bbc/" + folder + "/"
    # list each file in a particular category
    files = os.listdir(folder_path)
    for file_name in files:
        file_path = folder_path + "/" + file_name
        with open(file_path, errors = "replace") as text_file:
            title = text_file.readline()
            data = text_file.readlines()
        data = " ".join(data)
        title = title.strip()
        data = data.strip()
        # append to list of article titles
        article_titles.append(title)
        # append to list of article descriptions
        article_descriptions.append(data)
        # append to list of article categories
        article_categories.append(folder)
    
# create a dictionary of categories and descriptions
data_dict = {"title": article_titles, "description": article_descriptions, "category": article_categories}

# convert to csv file using pandas
df = pd.DataFrame(data_dict)
df.to_csv("./bbc_dataset.csv")