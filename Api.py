from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re
import warnings
import os
import gdown

# Suppress warnings
warnings.filterwarnings("ignore", message="The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.")

# Define the folder where the model will be saved
MODEL_DIR = "model"

# Google Drive folder ID (Extract from your shared link)
GDRIVE_FOLDER_ID = "1wjy2E0TCAZf92lwcvLRKTU5WG1qIKE3n"

# Function to download the model from Google Drive
def download_model():
    if not os.path.exists(MODEL_DIR):  # Download only if the model is not already present
        os.makedirs(MODEL_DIR)
        gdown.download_folder(id=GDRIVE_FOLDER_ID, output=MODEL_DIR, quiet=False)

# Load the model and tokenizer
def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model

def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer

# Initialize FastAPI app
app = FastAPI()

# Download and load the model
download_model()
model = load_model(MODEL_DIR)
tokenizer = load_tokenizer(MODEL_DIR)

# Define request structure
class QuestionRequest(BaseModel):
    question: str
    continue_generation: bool = False

def generate_text(model, tokenizer, sequence, max_length, continue_generation=False):
    # Encode input
    ids = tokenizer.encode(sequence, return_tensors='pt')
    attention_mask = (ids != tokenizer.pad_token_id).long()

    # Generate response
    final_outputs = model.generate(
        ids,
        attention_mask=attention_mask,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )

    # Process and clean output
    output_text = tokenizer.decode(final_outputs[0], skip_special_tokens=True)
    cleaned_answer = re.sub(r"\[Q\]|\[A\]", "", output_text).strip()

    return cleaned_answer

@app.post("/answer/")
async def get_answer(request: QuestionRequest):
    answer = generate_text(model, tokenizer, request.question, max_length=50, continue_generation=request.continue_generation)
    return {"answer": answer}
