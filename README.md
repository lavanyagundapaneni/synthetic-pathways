# Synthetic Academic Pathway Data Generation Using Amazon Bedrock
This document describes the process of generating synthetic academic pathway data for students in the German education system using Amazon Bedrock's LLaMA 3 model. The generated data includes details like current grade, school type, abitur score, future course, university, and more. This code aims to create realistic and logically consistent synthetic data entries.

## 1. Initializing the LLM:

Set up the Amazon Bedrock LLaMA 3 model using `boto3` for generating synthetic academic pathways.

```import boto3
import os
from dotenv import load_dotenv
import pandas as pd
import time
import re

# Load AWS credentials from .env file
load_dotenv()

# Initialize the Amazon Bedrock client
bedrock_client = boto3.client(
    service_name='bedrock',
    region_name=os.getenv('AWS_REGION'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)
```

## 2. Calling Bedrock to Generate Synthetic Data:

This function sends the prompt to Amazon Bedrock's LLaMA3 model and retrieves the response. It handles any potential errors that might occur during the API call.

```def call_bedrock(prompt):
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

```

## 3. Validating Responses:

This helper function checks if the response from the model contains all the necessary fields. This ensures that the generated data is in the correct format.

```
def is_valid_response(response):
    required_fields = [
        "Current Grade:", "School Type:", "Abitur Score:", "Future Course:",
        "University:", "Duration:", "Year:", "Category:", "Degree:",
        "Subject:", "Country:", "Financial Status:", "Stream:", "Curriculum:"
    ]
    return all(field in response for field in required_fields)
```

## 4. Generating Synthetic Data:

This function generates multiple synthetic academic pathway entries using the predefined prompt. It attempts to generate a valid response for each entry, retrying up to three times if the response format is incorrect.

```
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
```

## 5. Prompt Template: 
This template structures the synthetic academic pathway entries. Each entry generated by the LLaMA3 model follows this template to ensure consistency and realism.

```prompt_template = """
Current Grade: (choose different grades from 9th,10th,11th)
Future Course: (choose various courses and universities)
Duration: (vary between 2 to 4 years)
Year: (choose different years between 2025 to 2030)
Category: K12
School: (choose different schools)
Degree: (choose different degrees like Bachelors, Masters)
Subject: (choose different subjects like Economics, Computer Science, etc.)
University: (choose different universities)
Country: (choose different countries)
Financial Status: (choose budget in numericals)
Stream: (choose different streams like mpc,bipc,cec,hec etc...)
Curriculum: (choose different curriculum like cbse,ssc,etc.... )

Please provide the response in the following format:
Current Grade: ...
Future Course: ...
Duration: ...
Year: ...
Category: ...
School: ...
Degree: ...
Subject: ...
University: ...
Country: ...
Financial Status: ...
Stream: ...
Curriculum: ...
"""
```

## 6. Generating Entries:
The code generates the synthetic academic pathway entries by calling the `generate_synthetic_data` function using the prompt template.

```print("Generating synthetic entries...")
synthetic_entries = generate_synthetic_data(prompt_template, n_samples=20)
print("Generation complete.")
```

## 7. Parsing Generated Entries:
This function extracts the relevant fields from each generated entry. It splits the entry by lines and processes each field to remove unnecessary prefixes.

```def parse_entry(entry):
    lines = entry.strip().split('\n')
    values = []
    for line in lines:
        line = re.sub(r'^\d+\.\s*', '', line)
        if ": " in line:
            parts = line.split(": ", 1)
            if len(parts) == 2:
                values.append(parts[1].strip())
    return values

```

## 8. Structuring and Validating the Data:
The code parses the generated entries and ensures that each entry contains the expected number of fields. Invalid entries are filtered out.

```synthetic_data = []
for entry in synthetic_entries:
    parsed = parse_entry(entry)
    if len(parsed) == 150:
        synthetic_data.append(parsed)
    else:
        print(f"Invalid number of fields in entry: {entry}")
```



