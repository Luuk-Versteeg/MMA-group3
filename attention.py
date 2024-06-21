# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import torch
from transformers import pipeline

# pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto", output_attentions=True)
pipe = pipeline("text-generation", 
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                torch_dtype=torch.bfloat16, 
                device_map="auto"
            )

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "Can you ask me a question?"},
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, 
            #    top_p=0.95, output_attentions=True, output_hidden_states=True, return_dict_in_generate=False)
# # print(outputs[0]["generated_text"])
# # <|system|>
# # You are a friendly chatbot who always responds in the style of a pirate.</s>
# # <|user|>
# # How many helicopters can a human eat in one sitting?</s>
# # <|assistant|>
# # ...

input = pipe.preprocess(prompt)

with pipe.device_placement():
    inference_context = pipe.get_inference_context()
    with inference_context():
        model_inputs = pipe._ensure_tensor_on_device(input, device=pipe.device)
        # model_outputs = pipe._forward(model_inputs, **forward_params)
        model_outputs = pipe._forward(model_inputs)

        # input["input_ids"] = input["input_ids"].to('cuda')

        # OWN
        model_outputs = pipe.model.generate(input["input_ids"], 
                                            output_attentions=True, 
                                            output_hidden_states=True, 
                                            return_dict_in_generate=True)
        
        generated_sequence = model_outputs['sequences']

        in_b = input["input_ids"].shape[0]
        out_b = generated_sequence.shape[0]

        generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])


        pipe_out = {"generated_sequence": generated_sequence, "input_ids": input["input_ids"], "prompt_text": input["prompt_text"]}

        # pipe_out = pipe.forward(input, output_attentions=True, return_dict_in_generate=True)


        model_outputs = pipe._ensure_tensor_on_device(model_outputs, device=torch.device("cpu"))


out = pipe.postprocess(pipe_out)
# import pdb; pdb.set_trace()



def find_subsequence(main_seq, sub_seq):
    print("Find needle")
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

find = "Serve hot"

in_seq = input["input_ids"].squeeze()
out_seq = generated_sequence.squeeze()
needle = pipe.preprocess(find)["input_ids"].squeeze()

att = model_outputs["attentions"]
scores = [torch.cat(layer, dim=0) for layer in att]

import pdb; pdb.set_trace()

index = find_subsequence(out_seq, needle)

if index == -1:
    print("Couldn't find the needle...")
    exit()


# size [22,32,1,X-1]
score = scores[index - len(in_seq) - 1]

score = torch.mean(score, dim=[0,1]).squeeze()

previous_seq = out_seq[:index]

tokens = [pipe.tokenizer.decode(token) for token in previous_seq]

numpy_score = score.float().numpy()

[[token, sco] for token, sco in zip(tokens, numpy_score)]

att_text = pipe.tokenizer.decode(previous_seq)

# IMPORTANT: Generation script
# pipe.model.generate(pipe.preprocess(prompt)["input_ids"], output_attentions=True, max_time=15., do_sample=True, return_dict_in_generate=True)

# pipe.model.generate(pipe.preprocess(prompt)["input_ids"], output_attentions=True, return_dict_in_generate=True)

# from transformers import AutoModel, PreTrainedTokenizerFast

# model = AutoModel.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# tokenizer = PreTrainedTokenizerFast.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

import pdb; pdb.set_trace()