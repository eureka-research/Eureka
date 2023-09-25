# import requests

# API_URL = "https://api-inference.huggingface.co/models/WizardLM/WizardCoder-15B-V1.0"
# headers = {"Authorization": "Bearer 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"}

# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.json()
	
# output = query({
# 	"inputs": "Can you please let us know more details about your ",
# })
# print(output)

# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("WizardLM/WizardCoder-15B-V1.0")
# model = AutoModelForCausalLM.from_pretrained("WizardLM/WizardCoder-15B-V1.0")

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

# with torch.no_grad():
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
print(outputs)
from ipdb import set_trace
set_trace()