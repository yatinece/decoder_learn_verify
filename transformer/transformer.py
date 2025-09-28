import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as f
from config.ConfigFile import Config
from Tokenizer.BpeToken import BpeTokenizer
import math
from tqdm import tqdm

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def get_sequence_size(list_seq):
    return max(len(seq) for seq in list_seq)

def convert_listbatch_to_listtensor(batch_seq,device, max_length, dtype=torch.long):
    return [torch.tensor(seq[:max_length], device=device , dtype=dtype) for seq in batch_seq]

def pad_sequence_list(vector_list,max_length,pad_token_id=0):
    return torch.stack([f.pad(vector,(0,max_length-len(vector))) for vector in vector_list],dim=0)

def track_variance_batch(embeded_tensor):

    # 1. Overall Variance (across all elements)
    overall_variance = torch.var(embeded_tensor)
    print(f"Overall Variance: {overall_variance.item():.4f}")

    # 2. Variance Across the Embedding/Feature Dimension (dim=2)
    # Resulting shape: [10, 1759]
    variance_dim_2 = torch.var(embeded_tensor, dim=2)
    print(f"Mean Variance Across Features (dim=2): {variance_dim_2.mean().item():.4f}")

    # 3. Variance Across the Sequence/Length Dimension (dim=1)
    # Resulting shape: [10, 256]
    variance_dim_1 = torch.var(embeded_tensor, dim=1)
    print(f"Mean Variance Across Sequence (dim=1): {variance_dim_1.mean().item():.4f}")

    # 4. Variance Across the Batch Dimension (dim=0)
    # Resulting shape: [1759, 256]
    variance_dim_0 = torch.var(embeded_tensor, dim=0)
    print(f"Mean Variance Across Batch (dim=0): {variance_dim_0.mean().item():.4f}")

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(EmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_layer = nn.Embedding(num_embeddings=self.vocab_size , embedding_dim  =self.embed_dim )
        self.scale = math.sqrt(embed_dim)

    def forward(self, x):
        x = self.embed_layer(x)
        x = x * self.scale
        #print(embedded.shape)
        #print(f"embedded: {track_variance_batch(embedded)}")
        return x

class PositionEmbeddingLayer(nn.Module):
    def __init__(self, max_length, embed_dim):
        super(PositionEmbeddingLayer, self).__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.embed_layer = nn.Embedding(num_embeddings=self.max_length , embedding_dim  =self.embed_dim )

    def learnable_position_emb(self, x):
        seq_len = x.size(1)
        positon_num = torch.arange(0, seq_len, dtype=torch.long , device=x.device)  # torch.Size([max_length])
        positon_num = positon_num.unsqueeze(0) #  # torch.Size([1,max_length])
        positon_num = positon_num.expand(x.shape[0], seq_len)  # torch.Size([batch_size,max_length])
        return self.embed_layer(positon_num)


    def forward(self, x , type_pos_emb="learnable"):
        if type_pos_emb == "learnable":
            pos_embedded = self.learnable_position_emb(x)
        else:
            raise ValueError(f"Method {type_pos_emb} is not implemented")
        x = x + pos_embedded
        
        #print(embedded.shape)
        #print(f"embedded: {track_variance_batch(embedded)}")
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,embed_dim, num_heads ,dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        if embed_dim % num_heads == 0:
            print(f"Per Head the embedded are : {embed_dim / num_heads }")
        else :
            raise ValueError("Embedding_dimension should be divisible by num_heads ")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = int(embed_dim / num_heads)

        # embedding weight for Q, K, V
        self.q_linear = nn.Linear(embed_dim,embed_dim) #[hd,hd]
        self.k_linear = nn.Linear(embed_dim,embed_dim) #[hd,hd]
        self.v_linear = nn.Linear(embed_dim,embed_dim) #[hd,hd]

        self.out_linear = nn.Linear(embed_dim,embed_dim) #[hd,hd]
        self.dropout = nn.Dropout(dropout)
        self.scale =math.sqrt(self.head_dim)

    def transform_numhead(self,x):
        return x.view(x.shape[0],x.shape[1],self.num_heads,self.head_dim)  #  [b, t, h, d] 

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()

        #compute Q,K,V
        q_emb = self.q_linear(x) # [batch_size, seq_len, embed_dim] [b, t, hd] [hd,hd] = [b, t, hd] 
        k_emb = self.k_linear(x) # [batch_size, seq_len, embed_dim] [b, t, hd] [hd,hd] = [b, t, hd] 
        v_emb = self.v_linear(x) # [batch_size, seq_len, embed_dim] [b, t, hd] [hd,hd] = [b, t, hd] 

        q_emb = self.transform_numhead(q_emb).transpose(1,2)   #  [b, h,t, d] 
        k_emb = self.transform_numhead(k_emb).transpose(1,2)   #  [b, h,t, d] 
        v_emb = self.transform_numhead(v_emb).transpose(1,2)   #  [b, h,t, d] 


        Att = torch.matmul(q_emb, k_emb.transpose(2,3))  #  [b, h,t, d]  * [b, h, d ,t] =  [b, h,t, t]
        Att = Att /self.scale  # [b, h,t, t]

        if mask is not None:
            # Create a mask tensor with True/False (or 0/1) for valid/invalid entries
            Att = Att.masked_fill(mask == 0, float('-inf'))
        # Softmax to get attention weights
        Att = f.softmax(Att, dim=-1)
        Att = self.dropout(Att)

        # Compute attention output: (Att @ V)
        attn_output = torch.matmul(Att , v_emb ) #  [b, h,t, t] *  [b, h,t, d] =  [b, h,t, d]

        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size,seq_len ,embed_dim) # [b, h,t, d]= [b, t,h, d] = [b, t,hd]

        output_block_emb = self.out_linear(attn_output ) #  [b, t,hd] ==  [b, t,hd]

        return output_block_emb, Att


class TransformerDecoderBlock(nn.Module):
    def __init__(self,embed_dim, num_heads, ffn_dim, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.ffn_dim = ffn_dim
        self.attention_block = MultiHeadSelfAttention(self.embed_dim, self.num_heads , self.dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 =nn.LayerNorm(self.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.ffn_dim, self.embed_dim),
                    )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 =nn.LayerNorm(self.embed_dim)

    def forward(self, x , mask=None):
        output_block_emb, attention_weight = self.attention_block(x , mask)

        if output_block_emb.shape != x.shape:
            raise ValueError(f" Shape mismatch between input sequence embedding shape of : {x.shape} and attention block output shape :{output_block_emb.shape}")

        x = x + self.dropout1(output_block_emb)
        x = self.norm1(x)
        ffn_x = self.ffn(x)

        x = x + self.dropout2(ffn_x)
        x = self.norm2(x)

        return x, attention_weight


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ffn_dim, num_blocks , max_length, dropout=0.1, type_pos_emb="learnable"):
        super(TransformerDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.ffn_dim = ffn_dim
        self.num_blocks = num_blocks
        self.max_length = max_length
        self.type_pos_emb = type_pos_emb
        self.embedding_layer = EmbeddingLayer(self.vocab_size, self.embed_dim)
        self.pos_embedding_layer = PositionEmbeddingLayer(self.max_length, self.embed_dim)
        self.block_layers = nn.ModuleList(TransformerDecoderBlock(self.embed_dim, self.num_heads, self.ffn_dim, self.dropout) for _ in range(num_blocks) )
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask=None):
        batch_size, seq_len = x.size()
        if seq_len > self.max_length:
            raise ValueError(f"Sequence length {seq_len} exceeds max_length {self.max_length}") 

        attn_weights_list = []
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len ,device=x.device) )
            mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, seq_len, seq_len)

        x = self.embedding_layer(x)
        pos_embedded = self.pos_embedding_layer(x , self.type_pos_emb)
        x = self.dropout(x + pos_embedded)


        for block in self.block_layers:
            x, attention_weight = block(x , mask)
            attn_weights_list.append(attention_weight)
        output = self.fc_out(x)
        return output, attn_weights_list


if __name__ == "__main__":
    
    config_parser=Config()
    download_data_key = config_parser.download_data_key
    dataset = config_parser.dataset
    path = config_parser.path
    data_path= config_parser.data_path
    os.makedirs(data_path ,exist_ok=True)
    data= torch.load( data_path+"encoded_dataset_fast.pt")
    with open(path+'.meta.json' ) as json_data:
        meta_data_tokenizer=json.load(json_data)
    sequence_size = get_sequence_size(data)
    print(f"Max sequence size {sequence_size}")
    # Split data into train and validation
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]

    max_length=min(20000,sequence_size)

    batch_size=10
    vocab_size=meta_data_tokenizer["vocab_size"]
    embed_dim=256
    num_heads = 4
    dropout = 0.1
    ffn_dim = embed_dim*3
    num_blocks= 4
    type_pos_emb="learnable"
    print(f"Current stats for max_length : {max_length} , batch_size : {batch_size} , vocab_size : {vocab_size}  , data_length : {len(data)}")

    print(f"Current stats for num_embeddings : {vocab_size} , embedding_dim : {embed_dim} ")

    device= get_device()
    decoder_att = TransformerDecoder(vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads, \
                            ffn_dim = ffn_dim, num_blocks= num_blocks, max_length =max_length ,dropout=dropout , type_pos_emb=type_pos_emb).to(device)

    # optimizer 
    optimizer = torch.optim.Adam(decoder_att.parameters(), lr=0.001)
    # loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    best_val_loss = float('inf')
    epochs = 3
    checpoint_path = os.path.join(data_path, "best_run/")
    os.makedirs(checpoint_path , exist_ok=True)
    checkpoint_path = os.path.join(data_path, "best_model_dict.pt")
    full_model_path = os.path.join(data_path, "best_model_full.pt")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(epochs):
        decoder_att.train()
        train_loss = 0.0
        train_batches = math.ceil(len(train_data) / batch_size  )      
        pbar = tqdm(range(0 ,len(train_data),batch_size) , desc=f"Epoch {epoch+1} Training", total=train_batches)
        for k in pbar:
            batch = train_data[k:k+batch_size]
            #print(batch)
            vector_list = convert_listbatch_to_listtensor(batch, device, max_length, dtype=torch.long)
            #print(f"size of batch {[vector.shape for vector in vector_list]}")
            batch_tensor = pad_sequence_list(vector_list,max_length,pad_token_id=0)

            output, attn_weights_list = decoder_att(batch_tensor)

            loss = loss_fn(output[:, :-1, :].reshape(-1, vocab_size), batch_tensor[:, 1:].reshape(-1))
            pbar.set_postfix({'Batch Loss': f'{loss.item():.4f}'})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= train_batches
        train_perplexity = math.exp(train_loss)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Perplexity: {train_perplexity:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Validation loop
        decoder_att.eval()
        val_loss = 0.0
        val_batches = math.ceil(len(val_data) / batch_size  )      
        
        with torch.no_grad():
            pbar =tqdm(range(0 ,len(val_data),batch_size) , desc=f"Epoch {epoch+1} Validation", total=val_batches)
            for k in pbar:
                batch = val_data[k:k+batch_size]
                #print(batch)
                vector_list = convert_listbatch_to_listtensor(batch, device, max_length, dtype=torch.long)
                #print(f"size of batch {[vector.shape for vector in vector_list]}")
                batch_tensor = pad_sequence_list(vector_list,max_length,pad_token_id=0)

                output, _ = decoder_att(batch_tensor)

                loss = loss_fn(output[:, :-1, :].reshape(-1, vocab_size), batch_tensor[:, 1:].reshape(-1))
                pbar.set_postfix({'Batch Loss': f'{loss.item():.4f}'})
                val_loss += loss.item()
        val_loss /= val_batches
        val_perplexity = math.exp(val_loss)
        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.4f}")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(decoder_att , full_model_path)
            torch.save({'epoch': epoch,
                'model_state_dict': decoder_att.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved best model at epoch {epoch+1} with Val Loss: {val_loss:.4f}")

        scheduler.step()  # Update learning rate

            #print(f"size of batch tensor {[tensor.shape for tensor in batch_tensor]}")

            # ## create word embedding 
            # embed = nn.Embedding(num_embeddings=vocab_size , embedding_dim  =embedding_dim ).to(device)
            # embedded = embed(batch_tensor) * math.sqrt(embedding_dim)
            # print(embedded.shape)
            # print(f"embedded: {track_variance_batch(embedded)}")


            # ## create positional embedding
            # positional_embedding = nn.Embedding(num_embeddings=max_length,embedding_dim  =embedding_dim ).to(device)
            # positon_num = torch.arange(0, max_length, dtype=torch.long).to(device)  # torch.Size([max_length])
            # positon_num = positon_num.unsqueeze(0) #  # torch.Size([1,max_length])
            # positon_num = positon_num.expand(embedded.shape[0],max_length)  # torch.Size([batch_size,max_length])
            # pos_embedded = positional_embedding(positon_num)
            # print(pos_embedded.shape)
            # print(f"pos_embedded: {track_variance_batch(pos_embedded)}")

            # ## embedded + pos_embedded
            # pos_add_embedded = embedded + pos_embedded
            # print(pos_add_embedded.shape)
            # print(f"pos_add_embedded: {track_variance_batch(pos_add_embedded)}")

            
            # attn_output, attn_weights_list = decoder_att(batch_tensor, mask= True)
            # print(f"Attention output shape: {attn_output.shape}, dtype: {attn_output.dtype}")
            # print(f"Attention weights shape: {[attn_weights.shape for attn_weights in attn_weights_list]}")
            # print("Variance of attention output:")
            # track_variance_batch(attn_output)
            # print("*"*30)
            # print(f"Variance of attention weight: {[track_variance_batch(attn_weights) for attn_weights in attn_weights_list]}")
            
            # break   



