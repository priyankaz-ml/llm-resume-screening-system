import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


with open("sample_resume.pdf", "rb") as file:
    reader = PyPDF2.PdfReader(file)
    test=""
    for page in reader.pages:
        test += page.extract_text()

print("Resume Uploaded Successfully\n")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = text_splitter.split_text(test)

print(f"Created {len(chunks)} chunks\n")


#converting chunks into embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#storing the embeddings in chroma vector database
vectorstore = Chroma.from_texts(texts=chunks,embedding=embedding_model)

print("Embeddings stored in Chroma DB\n")

query = input("\nAsk a question about the resume: ")
if query.lower() == "exit":
    print("Goodbye!")
    exit()
results = vectorstore.similarity_search(query, k=2)

print("\nTop matching chunks\n")
for result in results:
   print(result.page_content)
   print("\n-----------------\n")


from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
context = "\n".join([result.page_content for result in results])

prompt = f""" Answer the question based on the following context below.

Context: {context}
Question: {query}
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

print("\nAI Answer:\n")
print(response.choices[0].message.content)