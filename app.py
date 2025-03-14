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
        "Full Name:", "Phone Number:", "Username:", "Country:",
        "State:", "City:", "Postal Code:", "Financial Situation:", "Grade:",
        "Grade Point Average:", "Curriculum:", "Stream:", "LinkedIn Profile:"
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


prompt_template = """Generate a synthetic user profile for a student pursuing higher education. 
Ensure that the data is realistic and aligns with real-world patterns. Each entry should be unique.

1. Full Name: (Generate the names from different countries )  
2. Phone Number: (Generate a valid, region-specific phone number)  
3. Username: (Generate a unique, natural-looking username)  
4. Country: (Choose a relevant country, e.g., USA, India, Germany, Canada)  
5. State: (Pick a realistic state based on the chosen country)  
6. City: (Generate a major or mid-sized city from the chosen state)  
7. Postal Code: (Generate a valid postal code for the city)  
8. Financial Situation: (Choose from: 0-25L, 25L-75L, 75L-3CR+)  
9. School Name: (Generate realistic school names based on the country)  
10. Grade: (Choose from 9th, 10th, 11th, 12th, considering typical school progress)  
11. Grade Point Average (GPA): (Vary between 0%-35%, 36%-60%, 61%-75%, 76%-85%, 86%-95%, 96%-100%)  
12. Curriculum: (Choose from CBSE, ICSE, IB, IGCSE, Nordic, etc.)  
13. Stream: (Choose from MPC, BIPC, CEC, HEC, MEC)  
14. LinkedIn Profile: (Generate a natural-looking LinkedIn URL format)  

Please provide the response in the following format:
Full Name: ...
Phone Number: ...
Username: ...
Country: ...
State: ...
City: ...
Postal Code: ...
Financial Situation: ...
School Name: ...
Grade: ...
Grade Point Average (GPA): ...
Curriculum: ...
Stream: ...
LinkedIn Profile: ...
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
