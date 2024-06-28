from sklearn import metrics
import torch
from transformers import pipeline
import numpy as np

snellius = False
model = True

if model:
    if snellius:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device=device) # For on Snellius
    else:
        model = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto") # For on local machine


def sent_classifier(prompt: str):   
    message = [
    {
        "role": "system",
        "content": "You classify sentences as possitive or negative.",
    },
    {"role": "user", "content": prompt},
    ]
    
    out, words, att_data = ask_model(message)

    answer = out[0]["generated_text"].split("<|assistant|>")[1]
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
    elif 'positive' in answer:
        predicted_label = "Positive"
    elif 'negative' in answer:
        predicted_label = "Negative"
    else:
        # unkown
        predicted_label = "Unknown"
       
    return predicted_label, words, att_data

def news_classifier(prompt: str):
    message = [
    {
        "role": "system",
        "content": "You classify sentences, you indicate in which part of the newspaper they appear, the possible sections are given to you by the prompt.",
    },
    {"role": "user", "content": prompt},
    ]

    out, words, att_data = ask_model(message)
    
    answer = out[0]["generated_text"].split("<|assistant|>")[1]
    answer = answer.lower()

    counts = {}
    counts['World'] = count_occurrences('world', answer)
    counts['Sports'] = count_occurrences('sport', answer)
    counts['Business'] = count_occurrences('business', answer)
    counts['Sci/Tech'] = count_occurrences('science', answer) + count_occurrences('technology', answer) + count_occurrences('sci/tech', answer)

    max_value = max(counts.values(), default=0)

    if max_value != 0:
        keys_with_max_value = [key for key in counts.keys() if counts[key] == max_value]
        if len(keys_with_max_value) == 1:
            predicted_label = keys_with_max_value[0]
        else:
            predicted_label = "Unknown"
    else:
        predicted_label = "Unknown"
    
    return predicted_label, words, att_data
    
def make_confusion_matrix(y_pred, y_true):
    return metrics.confusion_matrix(y_true, y_pred)

def count_occurrences(term: str, text: str) -> int:
    return text.count(term)

def ask_model(message, model=model):
    with torch.no_grad():
        prompt1 = model.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

        # CHANGE PARAMETERS IF NEEDED HERE
        out, seq_in, seq_out, scores = generate(model, prompt1, max_new_tokens=100, do_sample=False)

        words, att_matrix = process_attentions(seq_in, seq_out, scores, model)
        att_data = select_prompt_attention(words, att_matrix)
    
    return out, words, att_data


def generate(pipe, prompt, **generate_kwargs):

    input = pipe.preprocess(prompt)

    with pipe.device_placement():
        inference_context = pipe.get_inference_context()
        with inference_context():
            model_inputs = pipe._ensure_tensor_on_device(input, device=pipe.device)

            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs.get("attention_mask", None)
            # Allow empty prompts
            if input_ids.shape[1] == 0:
                input_ids = None
                attention_mask = None
                in_b = 1
            else:
                in_b = input_ids.shape[0]

            # If there is a prefix, we may need to adjust the generation length. Do so without permanently modifying
            # generate_kwargs, as some of the parameterization may come from the initialization of the pipeline.
            prefix_length = generate_kwargs.pop("prefix_length", 0)
            if prefix_length > 0:
                has_max_new_tokens = "max_new_tokens" in generate_kwargs or (
                    "generation_config" in generate_kwargs
                    and generate_kwargs["generation_config"].max_new_tokens is not None
                )
                if not has_max_new_tokens:
                    generate_kwargs["max_length"] = generate_kwargs.get("max_length") or pipe.model.config.max_length
                    generate_kwargs["max_length"] += prefix_length
                has_min_new_tokens = "min_new_tokens" in generate_kwargs or (
                    "generation_config" in generate_kwargs
                    and generate_kwargs["generation_config"].min_new_tokens is not None
                )
                if not has_min_new_tokens and "min_length" in generate_kwargs:
                    generate_kwargs["min_length"] += prefix_length

            model_outputs = pipe.model.generate(model_inputs["input_ids"], 
                                                output_attentions=True, 
                                                output_hidden_states=True, 
                                                return_dict_in_generate=True,
                                                attention_mask=attention_mask,
                                                **generate_kwargs)
            
            generated_sequence = model_outputs['sequences']

            in_b = model_inputs["input_ids"].shape[0]
            out_b = generated_sequence.shape[0]

            generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])
            pipe_out = {
                "generated_sequence": generated_sequence, 
                "input_ids": model_inputs["input_ids"], 
                "prompt_text": model_inputs["prompt_text"],
                "attentions": model_outputs["attentions"]
            }

            pipe_out = pipe._ensure_tensor_on_device(pipe_out, device=torch.device("cpu"))

    out = pipe.postprocess(pipe_out)

    att = pipe_out["attentions"]
    scores = [torch.cat(layer, dim=0) for layer in att]

    seq_in = pipe_out["input_ids"]
    seq_out = pipe_out["generated_sequence"]

    return out, seq_in, seq_out, scores


def process_attentions(seq_in, seq_out, scores, pipe):

    seq_in = seq_in.squeeze()
    seq_out = seq_out.squeeze()

    tokens = seq_out[len(seq_in):]
    tokens = [t for t in tokens.numpy()]

    d_in = len(seq_in)
    d_out = len(seq_out)

    words = []
    att_matrix = np.zeros((d_out, d_out))

    for index, (token, score) in enumerate(zip(tokens, scores)):

        word = pipe.tokenizer.decode(token)
        words.append(word)

        attention = score.mean(dim=[0,1]).float().numpy()

        if index == 0:
            att_matrix[:d_in, :d_in] = attention
        else:
            i = (d_in -1 + index)
            att_matrix[i:(i+ 1),:i+1] = attention

    words = [pipe.tokenizer.decode(t) for t in seq_in] + words

    string = pipe.tokenizer.decode(seq_out)
    spaces = [i for i,char in enumerate(string) if char == " "]

    count = 0
    spaces_placed = 0
    words_with_spaces = words.copy()

    for index, word in enumerate(words):

        count += len(word)

        if count+1 in spaces:
            words_with_spaces.insert(index+1+spaces_placed, " ")
            spaces.remove(count+1)
            spaces_placed += 1
            count += 1

    return words_with_spaces, att_matrix


def find_subsequence(main_seq, sub_seq):
    main_len = main_seq.size(0)
    sub_len = sub_seq.size(0)
    
    # Ensure the subsequence is not longer than the main sequence
    if sub_len > main_len:
        return -1  # or raise an exception or handle it as needed
    
    # Slide a window over the main sequence
    for i in range(main_len - sub_len + 1):
        # Check if the subsequence matches
        if torch.equal(main_seq[i:i + sub_len], sub_seq):
            return i  # Return the start index of the matching subsequence
    
    return -1  # If no match is found


def select_prompt_attention(words, att_matrix):

    att_data = []

    for index, word in enumerate(filter(lambda x: x != " ", words)):
        att_data.append([word, att_matrix[index].tolist()])

    return att_data
