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
