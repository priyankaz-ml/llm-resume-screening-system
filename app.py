from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=groq_api_key)

import PyPDF2

# -------- Extract text from PDF --------
with open("sample_resume.pdf", "rb") as file:
    reader = PyPDF2.PdfReader(file)

    text = ""

    for page in reader.pages:
        text += page.extract_text()


# -------- Skill extraction --------
skills_db = [
    "Python",
    "Machine Learning",
    "Deep Learning",
    "SQL",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "TensorFlow",
    "PyTorch",
    "Excel",
    "NLP",
    "Data Analysis",
    "C++"
]

found_skills = []

for skill in skills_db:
    if skill.lower() in text.lower():
        found_skills.append(skill)

print("Skills found in resume:")
print(found_skills)
# -------- Job Description --------

job_description = """
Looking for Machine Learning Intern with Python,
Machine Learning, SQL, TensorFlow and Pandas experience
"""


# -------- Extract skills from JD --------

jd_skills = []

for skill in skills_db:
    if skill.lower() in job_description.lower():
        jd_skills.append(skill)


# -------- Find common skills --------

matched_skills = []

for skill in found_skills:
    if skill in jd_skills:
        matched_skills.append(skill)


# -------- Calculate score --------

score = (len(matched_skills) / len(jd_skills)) * 100


print("\nJob Description Skills:")
print(jd_skills)

print("\nMatched Skills:")
print(matched_skills)

print(f"\nMatch Score: {score:.2f}%")

prompt = f"""
You are an AI recruiter.

Resume Skills:
{found_skills}

Job Description Skills:
{jd_skills}

Matched Skills:
{matched_skills}

Match Score:
{score:.2f}%

Give:

1. Candidate strengths
2. Missing skills
3. Short recruiter summary
4. Final recommendation:
5. A message for them to improve their resume if they are shortlisted or maybe, or a message to encourage them to apply for other roles if they are rejected.   
Shortlist / Maybe / Reject
"""

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nAI Recruiter Feedback:\n")
print(response.choices[0].message.content)