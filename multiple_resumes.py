import os
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

resume_folder = "resumes"

all_texts = []
all_metadata = []

for filename in os.listdir(resume_folder):
    if filename.endswith(".docx"):
        file_path = os.path.join(resume_folder,filename)
        doc = Document(file_path)
        text =""
        for para in doc.paragraphs:
            text+=para.text + "\n"
        all_texts.append(text)
        all_metadata.append({"candidate": filename})

print(f"\nLoaded {len(all_texts)} resumes\n")

# SPLIT INTO CHUNKS

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
all_chunks = []
all_chunk_metadata = []

for i in range(len(all_texts)):
    chunks = splitter.split_text(all_texts[i])
    for chunk in chunks:
        all_chunks.append(chunk)

        all_chunk_metadata.append(all_metadata[i])

print(f"Created {len(all_chunks)} chunks\n")

# CREATE EMBEDDINGS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# STORE IN CHROMA DB
vectorstore = Chroma.from_texts(texts=all_chunks,embedding=embedding_model,metadatas=all_chunk_metadata)
print("Embeddings stored in Chroma DB\n")

# ASK QUESTIONS
query = input("ask recruiter query: ")

results = vectorstore.similarity_search(query, k=5)

# SHOW RESULTS
print("\nTop matching candidates: \n")
for result in results:
    print(result.metadata)
    print(result.page_content[:300])
    print("\n---------------------------\n")