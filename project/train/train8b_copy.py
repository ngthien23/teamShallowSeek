import pickle
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from utils import recursive_embed_cluster_summarize, load_files_from_directory, embed

model = OllamaLLM(model="deepseek-r1:8b")
embd = OllamaEmbeddings(model='nomic-embed-text')

# Set the directory containing PDFs
directory = "..\data\ClimateChange"
docs_texts = load_files_from_directory(directory)
with open("docs_texts.pkl", "wb") as file:
    pickle.dump(docs_texts, file)

# Build tree
leaf_texts = docs_texts
results = recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=3)

with open("clusters.pkl", "wb") as file:
    pickle.dump(results, file)

# Initialize all_texts with leaf_texts
all_texts = leaf_texts.copy()

# Iterate through the results to extract summaries from each level and add them to all_texts
for level in sorted(results.keys()):
    # Extract summaries from the current level's DataFrame
    summaries = results[level][1]["summaries"].tolist()
    # Extend all_texts with the summaries from the current level
    all_texts.extend(summaries)

# Now, use all_texts to build the vectorstore with Chroma
vectorstore = Chroma.from_texts(texts=all_texts, embedding=embd, persist_directory="./chroma_db")