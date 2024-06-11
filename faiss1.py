import pandas as pd
pd.set_option('display.max_colwidth', 100)

df= pd.read_csv("sample_text.csv")
print(df.shape)
print(df)





from sentence_transformers import SentenceTransformer
encoder= SentenceTransformer("all-mpnet-base-v2")
vector= encoder.encode(df.text)
print(vector.shape)

dim=vector.shape[1]
print(dim)
import faiss

index= faiss.IndexFlatL2(dim)
index.add(vector)
print(index)
index.add(vector)


search_query = "I want to buy a polo t-shirt"
vec = encoder.encode(search_query)
vec.shape
print(vec.shape)

import numpy as np
svec = np.array(vec).reshape(1,-1)
print(svec.shape)

#step 5
distances, I = index.search(svec, k=2)
print(distances)
print(I)