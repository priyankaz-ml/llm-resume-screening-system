import os
from docx import Document

from dotenv import load_dotenv
from groq import Groq

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

#LOAD ENV

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# LOAD ALL RESUMES

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
        all_metadata.append({"candidate": filename, "resume_text": text})
print(f"\nLoaded {len(all_texts)} resumes\n")

#CHUNKING

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
all_chunks = []
all_chunk_metadata = [] 
for i in range(len(all_texts)):
    chunks = splitter.split_text(all_texts[i])
    for chunk in chunks:
        all_chunks.append(chunk)
        all_chunk_metadata.append(all_metadata[i])
print(f"Created {len(all_chunks)} chunks\n")

# EMBEDDINGS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# VECTOR DATABASE
vectorstore = Chroma.from_texts(texts=all_chunks,embedding=embedding_model,metadatas=all_chunk_metadata)
print("Vector DB created\n")

#JOB DECRIPTION
job_description = """
Looking for Java Developer

Required Skills:
Java
J2EE
Spring
SQL
REST API
"""
print("\nSearching Candidates....\n")

#RETRIEVAL
results = vectorstore.similarity_search(job_description, k=10)

#UNIQUE CANDIDATES
unique_candidates = {}
for result in results:
    candidate = result.metadata["candidate"]
    if candidate not in unique_candidates:
        unique_candidates[candidate] = (result.metadata["resume_text"])

print(
    f"Found {len(unique_candidates)} candidates\n"
)


# ====================================
# AI SCORING
# ====================================

for candidate, resume_text in unique_candidates.items():

    prompt = f"""
You are an expert recruiter.

Job Description:

{job_description}

Resume:

{resume_text}

Give:

1. Match Score out of 100

2. Matching Skills

3. Missing Skills

4. Recommendation

Format:

Score: XX

Matching Skills:
...

Missing Skills:
...

Recommendation:
...
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("\n========================")
    print(candidate)
    print("========================\n")

    print(
        response.choices[0].message.content
    )
