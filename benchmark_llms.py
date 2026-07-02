"""
Converts the raw dataset.txt (already comma-separated, with quoted
multi-line fields) into a proper customer_feedback.csv file.
"""

import sys
import csv
import pandas as pd
from google import genai
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import time
from dotenv import load_dotenv
import os
import json


if not load_dotenv():
    raise Exception("Error Loading env file")

Google_API_Key = os.environ['Google_API_key']
Groq_API_Key = os.environ['GROQ_API_KEY']
Groq_API_rate = {"opneai":{"input":0.00000015,"output":0.0000006}, "llama":{"input":0.00000059,"output":0.00000079}}
# Google_API_rate = {"opneai":{"input":0.0000003,"output":0.0000025}, "llama":{"input":0.0000003,"output":0.0000025}}

system_prompt = '''
                you are an expert in sentiment Analysis/classification, your task is to classify a sentence into 3 different sentiments below:
                    - 'Positive'
                    - 'Negative'
                    - 'Neutral'
                Don't include any extra text, reasoning statement and special characters etc. in your response, 
                Your response should include only one word among 3 sentiments mentioned above.
                \n\n\n
                '''

SOURCE_PATH = "dataset.txt"
OUTPUT_PATH = "customer_feedback.csv"
LABELS = ["Positive", "Negative", "Neutral"]


def creating_csv():

    with open(SOURCE_PATH, "r", encoding="utf-8",newline="") as infile:
        reader = csv.reader(infile)
        rows = list(reader)

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(rows)

# print(f"Converted {SOURCE_PATH} -> {OUTPUT_PATH} ({len(rows) - 1} data rows)")

def load_model(model_name):

    if model_name == "OpenAI":
        return ChatGroq(model = "openai/gpt-oss-120b",api_key = Groq_API_Key),Groq_API_rate["opneai"]
    else: return ChatGroq(model = "llama-3.3-70b-versatile",api_key = Groq_API_Key),Groq_API_rate['llama']


def call_models(df,model):

    model_response = []
    latency = []
    ip_tokens = []
    op_tokens = []

    for i in range(len(df)):
        query = "Evaluate the sentiment of the below feedback:\n" + df.iloc[i]['feedback_text']
        start_time = time.perf_counter()
        response = model.invoke(system_prompt + query)
        end_time = time.perf_counter()

        model_response.append(response.content)
        latency.append((end_time - start_time)*1000)
        ip_tokens.append(response.usage_metadata.get('input_tokens'))
        op_tokens.append(response.usage_metadata.get('output_tokens'))
    
    return model_response, latency, ip_tokens, op_tokens


def calculate_metrics(df,response,latency,ip_tokens,op_tokens,rate):

    match = 0
    confusion = {true: {pred: 0 for pred in LABELS + ["Unknown"]} for true in LABELS}

    print("Empty Confusion_Matrix:\n",confusion)

    for i in range(len(response)):
        true_label = df.iloc[i]['true_sentiment']
        pred_label = response[i].strip().capitalize()
        if pred_label not in LABELS:
            pred_label = "Unknown"

        confusion[true_label][pred_label] = confusion[true_label][pred_label] + 1

        if pred_label == true_label:
            match = match+1

    accuracy = match/len(response)
    latency = sum(latency)/len(latency)
    total_op_tokens = sum(op_tokens)
    total_ip_tokens = sum(ip_tokens)
    total_cost = (total_ip_tokens*rate['input']) + (total_op_tokens*rate['output'])

    return {"Overall_Accuracy":accuracy,"Average_Latency":latency,"Estimated_Total_Cost":total_cost,"Confusion_matrix":confusion}



def invoke():

    df = pd.read_csv("D:\Learning\Comapny_assignment\customer_feedback.csv")
    print(df.iloc[0]['feedback_text'])

    models = ["OpenAI","Lllama"]
    Model_reports = []

    for i in models:
        model,rate = load_model(i)
        response,latency,ip_tokens,op_tokens = call_models(df,model)
        Model_reports.append(calculate_metrics(df,response,latency,ip_tokens,op_tokens,rate))
    
    
    Final_report = {"openai/gpt-oss-120b":Model_reports[0], "llama-3.3-70b-versatile":Model_reports[1]}

    with open("benchmark_results.json", "w", encoding="utf-8") as file:
        json.dump(Final_report,file,indent=4)


if __name__=="__main__":
    invoke()
    







