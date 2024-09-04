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

Please provide the response in the following format:

1. Current Grade: 10th
   School Type: Gymnasium
   Abitur Score: 2.5
   Future Course: Computer Science
   University: Technical University of Munich
   Duration: 3 years
   Year: 2028
   Degree: Bachelor's
   Subject: Computer Science
   Country: Germany
   Stream: Naturwissenschaften (Natural Sciences)
   internships/praktikum:Summer coding bootcamp focused on software development
   extra activities: coding club,robotics etc..
   Aspirations:Aspires to work as a data scientist specializing in big data analytics

2. Current Grade: 11th
   School Type: Realschule
   Abitur Score: 3.8
   Future Course: Medicine
   University: Ludwig Maximilian University of Munich
   Duration: 5 years
   Year: 2030
   Degree: Bachelor's
   Subject: Medicine
   Country: Germany
   Stream: Medizin (Medicine)
   internships/praktikum:Completed a 3-month internship at a community health center focusing on general medicine
   extra activities:Active member of the school's Debate Club
   Aspirations:Aims to become a pediatrician specializing in child healthcare

Generate similar entries below:
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

