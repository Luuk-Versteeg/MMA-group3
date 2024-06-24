from sklearn import metrics
import torch
from transformers import pipeline

snellius = False
model = True

if model:
    if snellius:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device=device) # For on Snellius
    else:
        model = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto") # For on local machine

def sent_classifier(prompt: str):   
    messages = [
    {
        "role": "system",
        "content": "You classify sentences as possitive or negative.",
    },
    {f"role": "user", "content": prompt},
    ]
    
    with torch.no_grad():
        prompt1 = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = model(prompt1, max_new_tokens=100, do_sample=False)#, temperature=0.7, top_k=50, top_p=0.95)

    answer = outputs[0]["generated_text"].split("<|assistant|>")[1]
    answer = answer.lower()

    if 'positive' in answer and 'negative' in answer:
        # unkown
        positive_count = answer.count('positive')
        negative_count = answer.count('negative')
        if positive_count > negative_count:
            predicted_label = "Positive"
        elif negative_count > positive_count:
            predicted_label = "Negative"
        else:
            predicted_label = "Unknown"
        # print("beide -", answer)
    elif 'positive' in answer:
        predicted_label = "Positive"
    elif 'negative' in answer:
        predicted_label = "Negative"
    else:
        # unkown
        predicted_label = "Unknown"
        # print("else -", answer)
       
    return predicted_label, answer

def news_classifier(prompt: str):
    messages = [
    {
        "role": "system",
        "content": "You classify sentences, you indicate in which part of the newspaper they appear, the possible sections are given to you by the prompt.",
    },
    {f"role": "user", "content": prompt},
    ]
    
    with torch.no_grad():
        prompt1 = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = model(prompt1, max_new_tokens=200, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

    answer = outputs[0]["generated_text"].split("<|assistant|>")[1]
    answer = answer.lower()

    if 'world news' in answer:
        predicted_label = "World"
    elif 'sports' in answer or 'sport' in answer:
        predicted_label = "Sports"
    elif 'business' in answer:
        predicted_label = "Business"
    elif 'science' in answer or 'technology' in answer:
        predicted_label = "Sci/Tech"
    else:
        predicted_label = "Unknown"
    
    return predicted_label, answer
    
def make_confusion_matrix(y_pred, y_true):
    return metrics.confusion_matrix(y_true, y_pred)