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
    
    return predicted_label
    
def make_confusion_matrix(y_pred, y_true):
    return metrics.confusion_matrix(y_true, y_pred)


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
        # attention = score[0,0,:,:].float().numpy()

        if index == 0:
            att_matrix[:d_in, :d_in] = attention
        else:
            i = (d_in -1 + index)
            att_matrix[i:(i+ 1),:i+1] = attention

        # o.append([word, attention])

    words = [pipe.tokenizer.decode(t) for t in seq_in] + words

    return words, att_matrix


    # find = "Serve hot"
    # needle = pipe.preprocess(find)["input_ids"].squeeze()

    # in_seq = input["input_ids"].squeeze()
    # out_seq = generated_sequence.squeeze()


    # import pdb; pdb.set_trace()

    # index = find_subsequence(out_seq, needle)

    # if index == -1:
    #     print("Couldn't find the needle...")
    #     exit()


    # # size [22,32,1,X-1]
    # score = scores[index - len(in_seq) - 1]


    # previous_seq = out_seq[:index]

    # tokens = [pipe.tokenizer.decode(token) for token in previous_seq]

    # numpy_score = score.float().numpy()

    # [[token, sco] for token, sco in zip(tokens, numpy_score)]

    # att_text = pipe.tokenizer.decode(previous_seq)

    # IMPORTANT: Generation script
    # pipe.model.generate(pipe.preprocess(prompt)["input_ids"], output_attentions=True, max_time=15., do_sample=True, return_dict_in_generate=True)

    # pipe.model.generate(pipe.preprocess(prompt)["input_ids"], output_attentions=True, return_dict_in_generate=True)

    # from transformers import AutoModel, PreTrainedTokenizerFast

    # model = AutoModel.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # tokenizer = PreTrainedTokenizerFast.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # import pdb; pdb.set_trace()


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

    for index, word in enumerate(words):
        att_data.append([word, att_matrix[index].tolist()])

    return att_data



if __name__ == "__main__":

    prompt = "This is a test"

    messages = [
    {
        "role": "system",
        "content": "You classify sentences, you indicate in which part of the newspaper they appear, the possible sections are given to you by the prompt.",
    },
    {"role": "user", "content": prompt},
    ]
    
    with torch.no_grad():
        prompt1 = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # prompt1 = model.tokenizer.encode(prompt)
        out, seq_in, seq_out, scores = generate(model, prompt1, max_new_tokens=100, do_sample=False)

        words, att_matrix = process_attentions(seq_in, seq_out, scores, model)
        att_data = select_prompt_attention(words, att_matrix)

        import pdb; pdb.set_trace()

        # prompt_tokens = model.tokenizer.encode(prompt)


    import pdb; pdb.set_trace()