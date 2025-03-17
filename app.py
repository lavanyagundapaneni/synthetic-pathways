import pandas as pd
import os
import re
import openai
import json
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY

def call_openai(prompt):
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY) 
        response = client.chat.completions.create(
            model="gpt-4-turbo",  # Use the appropriate model version
            messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.5
        )                                                                                                                                                                             
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

def is_valid_response(response):
    required_fields = [
        "Path Name:", "Duration:", "Description:", "Type:",
        "Destination:", "Grade:", "Grade Point Average:", "Curriculum:", "Stream:", "Financial Requirement:", "Program Name:",
        "University City:", "University Country:"
    ]
    return all(field in response for field in required_fields)


def generate_synthetic_data(prompt, n_samples=20, max_retries=8):
    synthetic_data = []
    for i in range(n_samples):
        retries = 0
        while retries < max_retries:
            try:
                response = call_openai(prompt)
                if is_valid_response(response):
                    synthetic_data.append(response.strip())
                    break
                else:
                    print(f"Unexpected response format at sample {i + 1}: {response}")
                    retries += 1
            except Exception as e:
                print(f"Error at sample {i + 1}: {e}")
                retries += 1
        if retries == max_retries:
            print(f"Failed to generate valid response for sample {i + 1} after {max_retries} retries.")
    return synthetic_data

prompt_template = """Generate a synthetic step for the visa interview preparation career preparation plan for a student aiming for a specific university or career destination. Ensure that each step is realistic and follows an appropriate sequence leading to the final goal. Each entry should be unique.

1. Step Name: (Choose a relevant step such as  "Visa Interview Preparation,"  )  
2. Duration (Days): (Specify the number of days typically required for this step, e.g., 30 days, 90 days, 180 days)  
3. Macro View: (Provide a broad description of the initial services required for this step)  

Grade-wise Focus Areas:  
- Grade 9: (List the subjects and skills students should focus on in this grade)  
- Grade 10: (List the subjects and skills students should focus on in this grade)  
- Grade 11: (List the subjects and skills students should focus on in this grade)  
- Grade 12: (List the subjects and skills students should focus on in this grade)  

GPA-wise Recommendations:  
- 0%-35%: (Provide advice on improvement strategies, alternative career paths, or skill development)  
- 36%-60%: (Provide guidance on strengthening academic performance and extracurriculars)  
- 61%-75%: (Provide suggestions for improving scores and focusing on relevant subjects)  
- 76%-85%: (Highlight key areas to enhance further and preparation strategies)  
- 86%-95%: (Suggest ways to secure top-tier university admissions and scholarship opportunities)  
- 96%-100%: (Provide recommendations for maximizing success in competitive exams and top institutions)  

Curriculum-based Recommendations:  
- IB: (Describe how this curriculum aligns with the career path and any special requirements)  
- IGCSE: (Mention the key subjects and coursework relevant to the chosen path)  
- CBSE: (Provide insights on preparation and subject selection)  
- ICSE: (List the benefits and best practices for excelling in this curriculum)  
- Nordic: (Explain how this curriculum helps in reaching the final career goal)  

Stream-wise Recommendations:  
- MPC: (Describe how this stream prepares students for engineering, technology, etc.)  
- BIPC: (Provide insights on medical career preparation)  
- CEC: (Explain how commerce and economics can shape financial careers)  
- HEC: (Mention the relevance of humanities and social sciences in this path)  
- MEC: (Describe how this stream supports business, finance, and analytics careers)  

Financial Situation-based Planning:  
- 0-25L: (Provide recommendations for budget-friendly preparation strategies and scholarship options)  
- 25L-75L: (Suggest mid-range financial planning and optimal resource utilization)  
- 75L-3CR+: (Describe premium education plans, overseas study options, and elite university admissions)  
- Other: (Mention alternative funding sources, financial aid, and flexible education plans)  

Personality-based Recommendations (RIASEC Model):  
- Realistic: (Suggest careers that match practical, hands-on skills)  
- Investigative: (Recommend career paths that align with analytical and problem-solving abilities)  
- Artistic: (Provide insights into creative career choices)  
- Social: (Suggest education and career options that involve interpersonal interaction)  
- Enterprising: (Highlight entrepreneurial paths and business education)  
- Conventional: (List structured and organized career paths suitable for methodical thinkers)  

Please provide the response in the following format:

Step Name: ...  
Duration (Days): ...  
Macro View: ...  

Grade-wise Focus Areas:  
Grade 9: ...  
Grade 10: ...  
Grade 11: ...  
Grade 12: ...  

GPA-wise Recommendations:  
0%-35%: ...  
36%-60%: ...  
61%-75%: ...  
76%-85%: ...  
86%-95%: ...  
96%-100%: ...  

Curriculum-based Recommendations:  
IB: ...  
IGCSE: ...  
CBSE: ...  
ICSE: ...  
Nordic: ...  

Stream-wise Recommendations:  
MPC: ...  
BIPC: ...  
CEC: ...  
HEC: ...  
MEC: ...  

Financial Situation-based Planning:  
0-25L: ...  
25L-75L: ...  
75L-3CR+: ...  
Other: ...  

Personality-based Recommendations (RIASEC Model):  
Realistic: ...  
Investigative: ...  
Artistic: ...  
Social: ...  
Enterprising: ...  
Conventional: ...  

"""


print("Generating synthetic entries...")
synthetic_entries = generate_synthetic_data(prompt_template, n_samples=20)
print("Generation complete.")


def parse_entry(entry):
    lines = entry.strip().split('\n')
    values = []
    for line in lines:
        line = re.sub(r'^\d+\.\s*', '', line)
        if ": " in line:
            parts = line.split(": ", 1)
            if len(parts) == 2:
                values.append(parts[1].strip())
    return values

synthetic_data = []
for entry in synthetic_entries:
    parsed = parse_entry(entry)
    if len(parsed) == 14:
        synthetic_data.append(parsed)
    else:
        print(f"Invalid number of fields in entry: {entry}")
