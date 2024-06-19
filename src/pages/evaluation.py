# from .tinyllama import pipe as model
from sklearn import metrics
import torch
from transformers import pipeline

model = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

def sent_classifier(prompt: str, dataset, n_samples: int, shuffle: bool = True):
    if shuffle: dataset = dataset.shuffle()
    
    sentences = dataset["content"][:n_samples]
    labels = dataset["label"][:n_samples]
    predicted_labels = []
    for sentence in sentences:
        print("new sentence")
        prompt1 = prompt.replace("SENTENCE", sentence)
        messages = [
        {
            "role": "system",
            "content": "You classify sentences as possitive or negative.",
        },
        {f"role": "user", "content": prompt1},
        ]
        print("prompt prepare")
        prompt2 = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print("model input")
        outputs = model(prompt2, max_new_tokens=100, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        answer = outputs[0]["generated_text"].split("<|assistant|>")[1]

        answer = answer.lower()

        if 'positive' in answer and 'negative' in answer:
            # unknown
            predicted_labels.append(-1) 
        elif 'positive' in answer:
            # positive
            predicted_labels.append(1)
        elif 'negative' in answer:
            # negative
            predicted_labels.append(0)
        else:
            # unkown
            predicted_labels.append(-1) 
        print("label predicted")
    
    return predicted_labels, labels
    
def news_classifier(prompt: str, dataset, n_samples: int, shuffle: bool = True):
    if shuffle: dataset = dataset.shuffle()
    
    sentences = dataset["text"][:n_samples]
    labels = dataset["label"][:n_samples]
    predicted_labels = []
    for sentence in sentences:
        prompt1 = prompt.replace("SENTENCE", sentence)
        messages = [
        {
            "role": "system",
            "content": "You classify sentences, you indicate in which part of the newspaper they appear, the possible sections are given to you by the prompt.",
        },
        {f"role": "user", "content": prompt1},
        ]
        prompt2 = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = model(prompt2, max_new_tokens=200, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        answer = outputs[0]["generated_text"].split("<|assistant|>")[1]

        answer = answer.lower()

        if 'world news' in answer:
            predicted_labels.append(1)
        elif 'sports' in answer or 'sport' in answer:
            predicted_labels.append(2)
        elif 'business' in answer:
            predicted_labels.append(3)
        elif 'science' in answer or 'technology' in answer:
            predicted_labels.append(4)
        else:
            predicted_labels.append(-1)

    return predicted_labels, labels
    
def make_confusion_matrix(y_pred, y_true):
    return metrics.confusion_matrix(y_true, y_pred)