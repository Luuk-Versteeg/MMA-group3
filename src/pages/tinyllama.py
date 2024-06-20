from sklearn import metrics
import torch
from transformers import pipeline
# from tqdm import tqdme

print("Cuda available?:", torch.cuda.is_available())

torch.cuda.empty_cache()
model = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device='cuda')#device_map="auto")

# def sent_classifier(prompt: str, dataset, n_samples: int, shuffle: bool = True):
#     if shuffle: dataset = dataset.shuffle()
    
#     torch.cuda.empty_cache()

#     sentences = dataset["content"][:n_samples]
#     labels = dataset["label"][:n_samples]
#     predicted_labels = []
#     for sentence in tqdm(sentences, disable=False):
#         prompt1 = prompt.format(sentence=sentence)
#         messages = [
#         {
#             "role": "system",
#             "content": "You classify sentences as possitive or negative.",
#         },
#         {f"role": "user", "content": prompt1},
#         ]
#         prompt2 = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         with torch.no_grad():
#             outputs = model(prompt2, max_new_tokens=100, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

#         answer = outputs[0]["generated_text"].split("<|assistant|>")[1]
#         answer = answer.lower()

#         if 'positive' in answer and 'negative' in answer:
#             # unknown
#             predicted_labels.append(-1) 
#         elif 'positive' in answer:
#             # positive
#             predicted_labels.append(1)
#         elif 'negative' in answer:
#             # negative
#             predicted_labels.append(0)
#         else:
#             # unkown
#             predicted_labels.append(-1) 
    
#     torch.cuda.empty_cache()    
#     return predicted_labels, labels

def sent_classifier(prompt: str):    
    torch.cuda.empty_cache()

    messages = [
    {
        "role": "system",
        "content": "You classify sentences as possitive or negative.",
    },
    {f"role": "user", "content": prompt},
    ]
    prompt1 = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    with torch.no_grad():
        outputs = model(prompt1, max_new_tokens=100, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

    answer = outputs[0]["generated_text"].split("<|assistant|>")[1]
    answer = answer.lower()

    if 'positive' in answer and 'negative' in answer:
        # unkown
        predicted_label = "Unknown"
    elif 'positive' in answer:
        predicted_label = "Positive"
    elif 'negative' in answer:
        predicted_label = "Negative"
    else:
        # unkown
        predicted_label = "Unknown"
    
    torch.cuda.empty_cache()    
    return predicted_label
    
# def news_classifier(prompt: str, dataset, n_samples: int, shuffle: bool = True):
#     if shuffle: dataset = dataset.shuffle()
    
#     torch.cuda.empty_cache()

#     sentences = dataset["text"][:n_samples]
#     labels = dataset["label"][:n_samples]
#     predicted_labels = []
#     for sentence in sentences:
#         prompt1 = prompt.format(sentence=sentence)
#         messages = [
#         {
#             "role": "system",
#             "content": "You classify sentences, you indicate in which part of the newspaper they appear, the possible sections are given to you by the prompt.",
#         },
#         {f"role": "user", "content": prompt1},
#         ]
        
#         with torch.no_grad():
#             prompt2 = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#             outputs = model(prompt2, max_new_tokens=200, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

#         answer = outputs[0]["generated_text"].split("<|assistant|>")[1]

#         answer = answer.lower()

#         if 'world news' in answer:
#             predicted_labels.append(1)
#         elif 'sports' in answer or 'sport' in answer:
#             predicted_labels.append(2)
#         elif 'business' in answer:
#             predicted_labels.append(3)
#         elif 'science' in answer or 'technology' in answer:
#             predicted_labels.append(4)
#         else:
#             predicted_labels.append(-1)
    
#     torch.cuda.empty_cache()
#     return predicted_labels, labels

def news_classifier(prompt: str):
    torch.cuda.empty_cache()

    messages = [
    {
        "role": "system",
        "content": "You classify sentences, you indicate in which part of the newspaper they appear, the possible sections are given to you by the prompt.",
    },
    {f"role": "user", "content": prompt},
    ]
    
    prompt1 = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    with torch.no_grad():
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
    
    torch.cuda.empty_cache()
    return predicted_label
    
def make_confusion_matrix(y_pred, y_true):
    return metrics.confusion_matrix(y_true, y_pred)