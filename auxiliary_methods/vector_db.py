import chromadb
import openai
import os
import json


with open('/home/roee/.credentials/roee_credentials.json', 'r') as fp:
            openai.api_key = json.load(fp)['OPENAI_API_KEY']

def get_embedding(text, model="text-embedding-ada-002"):
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def create_embeddings_collection(docs):
    client = chromadb.Client()

    collection = client.create_collection("test")
    doc_embeddings = [get_embedding(doc) for doc in docs]
    collection.add(
    embeddings=doc_embeddings,
#     metadatas=[{}
#         {"uri": "img1.png", "style": "style1"},
#         {"uri": "img2.png", "style": "style2"},
#         {"uri": "img3.png", "style": "style1"},
#         {"uri": "img4.png", "style": "style1"},
#         {"uri": "img5.png", "style": "style1"},
#         {"uri": "img6.png", "style": "style1"},
#         {"uri": "img7.png", "style": "style1"},
#         {"uri": "img8.png", "style": "style1"},
#     ],
    documents=docs,
    ids=[f"{i}" for i in range(len(docs))],
    )
    return collection

def get_best_match(query, collection):
    query_embedding = get_embedding(query)
    query_result = collection.query(
        query_embeddings=query_embedding,
        n_results=1,
    )
    best_match = query_result['documents'][0][0]
    best_id = query_result['ids'][0][0]
    return best_id, best_match
    

# def test():
#     docs = ["how smart are you?", "what age is she?", "how tall is she?"]
#     query = "how old is she?"
#     collection = create_embeddings_collection(docs)
#     a = get_best_match(query, collection)

# if __name__=='__main__':
#     test()


