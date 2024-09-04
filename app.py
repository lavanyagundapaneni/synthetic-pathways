import pandas as pd
import os
import re
import boto3
import json
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')

bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def call_bedrock(prompt):
    try:
        native_request = {
            "prompt": prompt,
            "max_gen_len": 2048,
            "temperature": 0.5, 
        }
        response = bedrock_client.invoke_model(
            modelId='meta.llama3-70b-instruct-v1:0',
            body=json.dumps(native_request)
        )
        model_response = json.loads(response['body'].read())
        return model_response['generation']
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

def is_valid_response(response):
    required_fields = [
        "Current Grade:", "School Type:", "Abitur Score:", "Future Course:",
        "University:", "Duration:", "Year:", "Category:", "Degree:",
        "Subject:", "Country:", "Financial Status:", "Stream:", "Curriculum:"
    ]
    return all(field in response for field in required_fields)


def generate_synthetic_data(prompt, n_samples=20, max_retries=3):
    synthetic_data = []
    for i in range(n_samples):
        retries = 0
        while retries < max_retries:
            try:
                response = call_bedrock(prompt)
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


prompt_template = """Generate a synthetic academic pathway entry for a student in the German education system. Ensure each entry is unique and realistic.

1. Current Grade: (Choose different grades from 10th, 11th, 12th, considering the typical age range and school progress)
2. School Type: (Choose from Gymnasium, Realschule, Hauptschule, and consider possible transitions between school types based on performance)
3. Abitur Score: (Vary the score between 1.0 to 4.0, where 1.0 is the best. Ensure scores correlate with potential university courses and future opportunities)
4. Future Course: (Choose various courses like Engineering, Medicine, Law, Arts, Humanities, Sciences, Business Administration, etc., considering the student's background and interests)
5. University: (Choose different universities in Germany, like Ludwig Maximilian University of Munich, University of Heidelberg, Technical University of Munich, etc.)
6. Duration: (Vary between 3 to 5 years, depending on the degree and course complexity)
7. Year: (Choose different years between 2025 to 2030, taking into account the expected time of graduation and course duration)
8. Degree: (Choose different degrees like Bachelor, Master, or Dipl.-Ing., considering the future course selected)
9. Subject: (Choose different subjects like Economics, Computer Science, Mechanical Engineering, Philosophy, etc., aligning with the future course)
10. Country: Germany (This should be constant, as we're focusing on the German education system)
11. Stream: (Choose different streams like Naturwissenschaften (Natural Sciences), Geisteswissenschaften (Humanities), Wirtschaftswissenschaften (Economics), etc.)
12. Internships/Praktikum: (choose about any relevant internships or practical experience, including the field, duration, and role)
13. Aspirations: (Outline the student's career goals or academic ambitions, such as becoming a researcher, pursuing a managerial role, etc.)
14. Extracurricular Activities: (List any extracurricular activities, such as robotics clubs, coding clubs, online courses, and internships in tech, that the student is involved in)
Please provide the response in the following format:
Current Grade: ...
School Type: ...
Abitur Score: ...
Future Course: ...
University: ...
Duration: ...
Year: ...
Degree: ...
Subject: ...
Country: Germany
Stream: ...
Internships/Praktikum: ...
Aspirations: ...
Extracurricular Activities: ...
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
    if len(parsed) == 150:
        synthetic_data.append(parsed)
    else:
        print(f"Invalid number of fields in entry: {entry}")

columns = [
    "Current Grade", "School Type", "Abitur Score", "Future Course",
    "University", "Duration", "Year", "Category", "Degree",
    "Subject", "Country", "Financial Status", "Stream", "Curriculum"
]

