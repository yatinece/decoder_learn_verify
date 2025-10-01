from Tokenizer.BpeToken  import BpeTokenizer
from config.ConfigFile import Config
import torch
import torch.nn as nn
import os
from transformer.transformer import *
import time
# read config value
config = Config()
print(dict(config.__dict__.items()))
sample_method = "argmax"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load model 

model=torch.load(os.path.join(config.data_path,"best_model_full.pt") , weights_only=False)
model.to(device) 
time.sleep(1)
tokenizer=BpeTokenizer.load(config.path)
eos_token_id = tokenizer.tokenizer.token_to_id("<EOS>")
#print(count_parameters(model, detailed=True))


input_text=""
while 1==1:
    print("Enter Text:")
    if input_text != "exit":
        input_text = input()

        max_new_tokens=5
        token_data =tokenizer.encode(input_text)
        print(f" The text {input_text} has been converted to {token_data}")

        print("Start to inference")
        model.eval()
        with torch.no_grad():
            token_data =tokenizer.encode(input_text)
            if eos_token_id is not None and token_data[-1] == eos_token_id:
                token_data = token_data[:-1]
            token_tensor_list_raw = token_data.copy()
            #token_tensor_list = pad_sequence_list(token_tensor_list,model.max_length,pad_token_id=0)
            print(f"token_data is {token_data}")
            for _ in tqdm(range(max_new_tokens), desc="Generating Tokens", total=max_new_tokens):
                token_tensor_list_nopad = convert_listbatch_to_listtensor([token_data] , device, max_length=model.max_length, dtype=torch.long)
                if len(token_data) < model.max_length:
                    token_tensor_list = pad_sequence_list(token_tensor_list_nopad,model.max_length,pad_token_id=0)
                else :
                    token_tensor_list = token_tensor_list_nopad[:,-model.max_length:]
                print(f"token_data is {token_data}")
                output, _ = model(token_tensor_list)
                new_token_logit = output[:,-1:,:]

                new_token_prob=nn.functional.softmax(new_token_logit,dim=-1)
                if sample_method == "argmax":
                    next_token = torch.argmax(new_token_prob, dim=-1, keepdim=True).squeeze(1)
                else:
                    next_token = torch.multinomial(new_token_prob, num_samples=1)
                #print(f"next_token is {next_token}")
                ##print(f"Shape of token_tensor_list is {token_tensor_list.shape}")
                #print(f"Shape of next_token is {next_token.shape}")
                next_token_id = next_token.item()


                token_data.append(next_token_id)
                if eos_token_id is not None and next_token_id == eos_token_id:
                    print("EOS token generated. Stopping early.")
                    break
                #token_tensor_list = token_tensor_list[:,1:]
                
        output = tokenizer.decode(token_data,skip_special_tokens=False).rstrip()

        print(f" Output is {output}")
    else:
        exit

