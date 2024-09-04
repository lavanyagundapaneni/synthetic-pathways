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

## 2. Generating Synthetic Data:
Use a predefined prompt template to generate synthetic entries. This function manages error handling and ensures each entry is formatted correctly.

```def generate_synthetic_data(prompt, n_samples, max_gen_len, temperature):
    synthetic_data = []
    
    for i in range(n_samples):
        try:
            response = bedrock_client.invoke_model(
                modelId="anthropic.claude-v2",
                body={
                    "prompt": prompt,
                    "max_tokens_to_sample": max_gen_len,
                    "temperature": temperature
                }
            )
            response_text = response.get("body", "").strip()

            if "Current Grade:" in response_text and "Future Course:" in response_text:
                synthetic_data.append(response_text)
            else:
                print(f"Unexpected response format at sample {i + 1}: {response_text}")
                
        except Exception as e:
            print(f"Error at sample {i + 1}: {e}")
            time.sleep(1)  # Adding a delay in case of an error to avoid rapid retries
    
    return synthetic_data
```

## 3. Prompt Template: 
Define the structure of the synthetic academic pathway entries, ensuring each generated entry is well-defined and unique.

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

