import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as f
from config.ConfigFile import Config
from Tokenizer.BpeToken import BpeTokenizer
import math
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import wandb

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

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(EmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)

    def forward(self, x):
        return self.embed_layer(x)

class PositionEmbeddingLayer(nn.Module):
    def __init__(self, max_length, embed_dim):
        super(PositionEmbeddingLayer, self).__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.embed_layer = nn.Embedding(num_embeddings=self.max_length, embedding_dim=self.embed_dim)

    def forward(self, x):
        # x: [batch, seq, d_model] (already token-embedded)
        seq_len = x.size(1)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(x.size(0), seq_len)
        pos_embedded = self.embed_layer(position_ids)
        return x + pos_embedded

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,embed_dim, num_heads ,dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        if embed_dim % num_heads == 0:
            print(f"Per Head the embedded are : {embed_dim / num_heads }")
        else:
            raise ValueError("Embedding_dimension should be divisible by num_heads ")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = int(embed_dim / num_heads)

        self.q_linear = nn.Linear(embed_dim,embed_dim)
        self.k_linear = nn.Linear(embed_dim,embed_dim)
        self.v_linear = nn.Linear(embed_dim,embed_dim)
        self.out_linear = nn.Linear(embed_dim,embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale =math.sqrt(self.head_dim)

    def transform_numhead(self,x):
        return x.view(x.shape[0],x.shape[1],self.num_heads,self.head_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()

        q_emb = self.q_linear(x)
        k_emb = self.k_linear(x)
        v_emb = self.v_linear(x)

        q_emb = self.transform_numhead(q_emb).transpose(1,2)
        k_emb = self.transform_numhead(k_emb).transpose(1,2)
        v_emb = self.transform_numhead(v_emb).transpose(1,2)

        Att = torch.matmul(q_emb, k_emb.transpose(2,3))
        Att = Att /self.scale

        if mask is not None:
            Att = Att.masked_fill(mask == 0, float('-inf'))
        
        Att = f.softmax(Att, dim=-1)
        Att = self.dropout(Att)

        attn_output = torch.matmul(Att, v_emb)
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size,seq_len,embed_dim)
        output_block_emb = self.out_linear(attn_output)

        return output_block_emb, Att


class TransformerDecoderBlock(nn.Module):
    def __init__(self,embed_dim, num_heads, ffn_dim, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.ffn_dim = ffn_dim
        self.attention_block = MultiHeadSelfAttention(self.embed_dim, self.num_heads, self.dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.ffn_dim, self.embed_dim),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(self.embed_dim)

    def forward(self, x, mask=None):
        output_block_emb, attention_weight = self.attention_block(x, mask)

        x = x + self.dropout1(output_block_emb)
        x = self.norm1(x)
        ffn_x = self.ffn(x)

        x = x + self.dropout2(ffn_x)
        x = self.norm2(x)

        return x, attention_weight


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ffn_dim, num_blocks, max_length, dropout=0.1, tied_emb=0):
        super(TransformerDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.ffn_dim = ffn_dim
        self.num_blocks = num_blocks
        self.max_length = max_length
        
        self.embedding_layer = EmbeddingLayer(self.vocab_size, self.embed_dim)
        self.pos_embedding_layer = PositionEmbeddingLayer(self.max_length, self.embed_dim)
        self.block_layers = nn.ModuleList([
            TransformerDecoderBlock(self.embed_dim, self.num_heads, self.ffn_dim, self.dropout_rate) 
            for _ in range(num_blocks)
        ])
        self.fc_out = nn.Linear(embed_dim, vocab_size, bias=False)
        
        if tied_emb:
            self.fc_out.weight = self.embedding_layer.embed_layer.weight
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len = x.size()
        if seq_len > self.max_length:
            raise ValueError(f"Sequence length {seq_len} exceeds max_length {self.max_length}")

        attn_weights_list = []
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
            mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, seq_len, seq_len)

        x = self.embedding_layer(x)
        x = self.pos_embedding_layer(x)
        x = self.dropout(x)

        for block in self.block_layers:
            x, attention_weight = block(x, mask)
            attn_weights_list.append(attention_weight)
        
        output = self.fc_out(x)
        return output, attn_weights_list

def count_parameters(model, detailed=True):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    
    if detailed:
        print("\nDetailed parameter breakdown:")
        print("-" * 50)
        for name, module in model.named_modules():
            if isinstance(module, (nn.Embedding, nn.Linear, nn.LayerNorm)):
                module_params = sum(p.numel() for p in module.parameters())
                print(f"{name}: {module_params:,} parameters")
        print("-" * 50)
    
    if hasattr(model, 'fc_out') and hasattr(model, 'embedding_layer'):
        tied = id(model.fc_out.weight) == id(model.embedding_layer.embed_layer.weight)
        print(f"Word embedding and fc_out weights are {'tied' if tied else 'not tied'}")
    
    return total_params

if __name__ == "__main__":
    
    config_parser = Config()
    dataset = config_parser.dataset
    path = config_parser.path
    data_path = config_parser.data_path
    os.makedirs(data_path, exist_ok=True)
    
    data = torch.load(data_path + "encoded_dataset_long.pt")
    with open(path + '.meta.json') as json_data:
        meta_data_tokenizer = json.load(json_data)
    
    sequence_size = get_sequence_size(data)
    print(f"Max sequence size {sequence_size}")
    
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]

    max_length = config_parser.max_length

    # CORRECTED HYPERPARAMETERS
    batch_size = 16
    vocab_size = meta_data_tokenizer["vocab_size"]
    embed_dim = 768
    num_heads = 12
    dropout = 0.1
    ffn_dim = embed_dim * 4
    num_blocks = 12
    lr = 8e-4  
    epochs = 128
    tied_emb = 1
    grad_clip_max_norm = 1.0
    opt_meth = "linear+cos"
    weight_decay = 0.01
    warmup_ratio = 0.15
    accumulation_steps = 8
    
    print(f"Current stats for max_length : {max_length} , batch_size : {batch_size} , vocab_size : {vocab_size}  , data_length : {len(data)}")
    print(f"Effective batch size (with accumulation): {batch_size * accumulation_steps}")
    
    wandb.init(project="transformer_training", name="run_transformer_fixed", config={
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "ffn_dim": ffn_dim,
        "num_blocks": num_blocks,
        "max_length": max_length,
        "batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_size * accumulation_steps,
        "epochs": epochs,
        "learning_rate": lr,
        "dropout": dropout,
        "tied_emb": tied_emb,
        "grad_clip_max_norm": grad_clip_max_norm,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "opt_meth": opt_meth
    })
    
    device = get_device()
    decoder_att = TransformerDecoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_blocks=num_blocks,
        max_length=max_length,
        dropout=dropout,
        tied_emb=tied_emb
    ).to(device)
    
    total_params = count_parameters(decoder_att, detailed=True)
    wandb.config.update({"total_parameters": total_params})

    steps_per_epoch = math.ceil(len(train_data) / batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(warmup_ratio * total_steps)
    
    print(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    optimizer = torch.optim.AdamW(decoder_att.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    best_val_loss = float('inf')
    
    checkpoint_path = os.path.join(data_path, "best_model_dict.pt")
    full_model_path = os.path.join(data_path, "best_model_full.pt")
    
    # FIXED: Correct scheduler without multiplying by accumulation_steps
    def lr_lambda(current_step):
        # current_step is already the optimizer step count
        act_step = accumulation_steps*current_step
        if act_step < warmup_steps:
            return float(act_step) / max(1, warmup_steps)
        progress = float(act_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = LambdaLR(optimizer, lr_lambda)

    global_step = 0  # Tracks optimizer steps, not batch steps
    optimizer.zero_grad()
    
    for epoch in range(epochs):
        decoder_att.train()
        train_loss = 0.0
        train_batches = math.ceil(len(train_data) / batch_size)
        batch_count = 0
        
        pbar = tqdm(range(0, len(train_data), batch_size), desc=f"Epoch {epoch+1} Training", total=train_batches)
        
        for k in pbar:
            batch = train_data[k:k+batch_size]
            vector_list = convert_listbatch_to_listtensor(batch, device, max_length, dtype=torch.long)
            batch_tensor = pad_sequence_list(vector_list, max_length, pad_token_id=0)

            output, attn_weights_list = decoder_att(batch_tensor)
            loss = loss_fn(output[:, :-1, :].reshape(-1, vocab_size), batch_tensor[:, 1:].reshape(-1))
            
            # Gradient accumulation
            loss_for_backward = loss / accumulation_steps
            loss_for_backward.backward()
            
            batch_count += 1
            should_step = (batch_count % accumulation_steps == 0)
            
            if should_step:
                torch.nn.utils.clip_grad_norm_(decoder_att.parameters(), max_norm=grad_clip_max_norm)
                
                grad_norm_total = torch.sqrt(sum(p.grad.norm(2) ** 2 for p in decoder_att.parameters() if p.grad is not None))
                
                # Log at optimizer step
                wandb.log({
                    "Train/Batch_Loss": loss.item(),
                    "Train/Grad_Norm_Total": grad_norm_total.item(),
                    "Learning_Rate": optimizer.param_groups[0]['lr']
                }, step=global_step)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1  # FIXED: Only increment when optimizer steps
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.6f}',
                    'Step': global_step
                })
            else:
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Accum': f'{batch_count % accumulation_steps}/{accumulation_steps}'})
            
            train_loss += loss.item()
        
        # Handle remaining gradients
        if batch_count % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(decoder_att.parameters(), max_norm=grad_clip_max_norm)
            grad_norm_total = torch.sqrt(sum(p.grad.norm(2) ** 2 for p in decoder_att.parameters() if p.grad is not None))
            
            wandb.log({
                "Train/Batch_Loss": loss.item(),
                "Train/Grad_Norm_Total": grad_norm_total.item(),
                "Learning_Rate": optimizer.param_groups[0]['lr']
            }, step=global_step)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
        
        train_loss /= train_batches
        train_perplexity = math.exp(min(train_loss, 20))
        
        wandb.log({
            "Train/Epoch_Loss": train_loss,
            "Train/Perplexity": train_perplexity,
        }, step=global_step)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Perplexity: {train_perplexity:.2f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}, Global Step: {global_step}")

        # FIXED: Validation loop without global_step increment
        decoder_att.eval()
        val_loss = 0.0
        val_batches = math.ceil(len(val_data) / batch_size)
        
        with torch.no_grad():
            pbar = tqdm(range(0, len(val_data), batch_size), desc=f"Epoch {epoch+1} Validation", total=val_batches)
            for k in pbar:
                batch = val_data[k:k+batch_size]
                vector_list = convert_listbatch_to_listtensor(batch, device, max_length, dtype=torch.long)
                batch_tensor = pad_sequence_list(vector_list, max_length, pad_token_id=0)

                output, _ = decoder_att(batch_tensor)
                loss = loss_fn(output[:, :-1, :].reshape(-1, vocab_size), batch_tensor[:, 1:].reshape(-1))
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                val_loss += loss.item()
                # FIXED: No global_step increment during validation
            
            val_loss /= val_batches
            val_perplexity = math.exp(min(val_loss, 20))
            
            wandb.log({
                "Val/Epoch_Loss": val_loss,
                "Val/Perplexity": val_perplexity
            }, step=global_step)
            
            print(f"  Val Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.2f}")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(decoder_att, full_model_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': decoder_att.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'global_step': global_step
            }, checkpoint_path)
            print(f"  ✓ Saved best model (Val Loss: {val_loss:.4f})")

    wandb.finish()
    print("\nTraining completed!")