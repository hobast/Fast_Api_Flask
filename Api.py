from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", message="The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.")

# Load the model and tokenizer from the provided paths
def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model

def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    
    # If the tokenizer doesn't have a pad_token_id, we can set it to eos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer

# FastAPI app initialization
app = FastAPI()

# Model path (replace with your model's path)
model_path = "D:\\chat Api\\Model"  # Ensure this is correct
model = load_model(model_path)
tokenizer = load_tokenizer(model_path)

# Define the input structure
class QuestionRequest(BaseModel):
    question: str
    continue_generation: bool = False  # To check if user wants to continue the response

def generate_text(model, tokenizer, sequence, max_length, continue_generation=False):
    # Encode the question
    ids = tokenizer.encode(sequence, return_tensors='pt')

    # Create attention mask (1 for actual tokens, 0 for padding tokens)
    attention_mask = (ids != tokenizer.pad_token_id).long()  # Ensure it's a tensor

    # Generate the model's response
    final_outputs = model.generate(
        ids,
        attention_mask=attention_mask,  # Pass the attention mask here
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )

    # Decode the output and clean it
    output_text = tokenizer.decode(final_outputs[0], skip_special_tokens=True)

    # Step 1: Remove the [Q] and [A] tags (if present)
    cleaned_answer = re.sub(r"\[Q\]|\[A\]", "", output_text)

    # Step 2: Remove any question fragments, but keep the period (.)
    # This regex will remove question fragments but ensure the period is retained.
    cleaned_answer = re.sub(r'(\s*\?)(?=\s*$)', '.', cleaned_answer)

    # Step 3: Remove everything before and including the first newline
    cleaned_answer = cleaned_answer.split('\n', 1)[-1]

    # Step 4: Ensure there are no extra newlines between sentences
    cleaned_answer = re.sub(r'\n+', ' ', cleaned_answer)

    # Step 5: Split the answer into sentences, keeping only those that end with a period
    sentences = re.split(r'(?<=[.!?]) +', cleaned_answer)  # Split by sentence-ending punctuation followed by space
    complete_sentences = [sentence for sentence in sentences if sentence.endswith('.')]  # Keep only complete sentences

    # Join the complete sentences back together
    cleaned_answer = ' '.join(complete_sentences)

    return cleaned_answer.strip()

@app.post("/answer/") 
async def get_answer(request: QuestionRequest):
    # Get the question from the request
    question = request.question
    continue_generation = request.continue_generation

    # Generate the answer using the fine-tuned model
    answer = generate_text(model, tokenizer, question, max_length=50, continue_generation=continue_generation)

    # Return the generated answer as a JSON response
    return {"answer": answer}
