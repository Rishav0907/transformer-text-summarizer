import torch
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer
from collections import defaultdict
from tqdm import tqdm
import sys
import os
import socket
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Bloc.transformer import Transformer
from configs.config import CONFIG

# Hyperparameters
BATCH_SIZE = CONFIG["batch_size"]
HIDDEN_DIMS = CONFIG["hidden_dims"]
NUM_ENCODER_LAYERS = CONFIG["num_encoder_layers"]
NUM_DECODER_LAYERS = CONFIG["num_decoder_layers"]
EPOCHS = CONFIG["num_epochs"]
LEARNING_RATE = CONFIG["learning_rate"]

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = socket.gethostbyname(socket.gethostname())
    port = get_free_port()
    os.environ['MASTER_PORT'] = str(port)
    
    # Initialize the process group
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Explicitly set the device
    torch.cuda.set_device(rank)

def cleanup():
    torch.distributed.destroy_process_group()

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sos_idx = tokenizer.cls_token_id
eos_idx = tokenizer.sep_token_id
pad_idx = tokenizer.pad_token_id

# Load CNN/Daily Mail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Preprocessing function
def preprocess_data(examples):
    inputs = [text for text in examples['article']]
    targets = [summary for summary in examples['highlights']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length", add_special_tokens=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = dataset["train"].select(range(6000))
val_dataset = dataset["validation"].select(range(1000))
test_dataset = dataset["test"].select(range(100))

train_dataset = train_dataset.map(preprocess_data, batched=True, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(preprocess_data, batched=True, remove_columns=val_dataset.column_names)
test_dataset = test_dataset.map(preprocess_data, batched=True, remove_columns=test_dataset.column_names)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

def train(rank, world_size):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)

    # Create model and move it to GPU with id rank
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = Transformer(HIDDEN_DIMS, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, len(tokenizer), sos_idx, eos_idx, pad_idx, device).to(device)
    ddp_model = DDP(model, device_ids=[rank])

    # Prepare data loaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,num_workers=4,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler,num_workers=4,pin_memory=True)

    optimizer = optim.Adam(ddp_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Training function
    def train_epoch(model, train_loader, optimizer, criterion):
        model.train()
        total_loss = 0
        total_rouge = defaultdict(float)
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        progress_bar = tqdm(train_loader, desc=f"Training (GPU {rank})")
        for batch in progress_bar:
            inputs = batch['input_ids'].to(device)
            targets = batch['labels'].to(device)
            decoder_input = targets[:, :-1].to(device)  # Exclude the last token (EOS)
            targets_output = targets[:, 1:].to(device)
            optimizer.zero_grad()
            outputs = model(inputs, decoder_input)

            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets_output.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted_summaries = outputs.argmax(dim=-1)

            for pred, target in zip(predicted_summaries, targets):
                pred_decoded = tokenizer.decode(pred, skip_special_tokens=True)
                target_decoded = tokenizer.decode(target, skip_special_tokens=True)
                scores = scorer.score(target_decoded, pred_decoded)
                for metric, score in scores.items():
                    total_rouge[metric] += score.fmeasure

            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        avg_rouge = {metric: score / len(train_loader.dataset) for metric, score in total_rouge.items()}
        return avg_loss, avg_rouge

    # Validation function
    def validate(model, val_loader, criterion):
        model.eval()
        total_loss = 0
        total_rouge = defaultdict(float)
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        progress_bar = tqdm(val_loader, desc=f"Validating (GPU {rank})")
        with torch.no_grad():
            for batch in progress_bar:
                inputs = batch['input_ids'].to(device)
                targets = batch['labels'].to(device)
                decoder_input = targets[:, :-1].to(device)  # Exclude the last token (EOS)
                targets_output = targets[:, 1:].to(device)
                outputs = model(inputs, decoder_input)

                loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets_output.reshape(-1))
                total_loss += loss.item()
                predicted_summaries = outputs.argmax(dim=-1)

                for pred, target in zip(predicted_summaries, targets):
                    pred_decoded = tokenizer.decode(pred, skip_special_tokens=True)
                    target_decoded = tokenizer.decode(target, skip_special_tokens=True)
                    scores = scorer.score(target_decoded, pred_decoded)
                    for metric, score in scores.items():
                        total_rouge[metric] += score.fmeasure

                progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(val_loader)
        avg_rouge = {metric: score / len(val_loader.dataset) for metric, score in total_rouge.items()}
        return avg_loss, avg_rouge

    # Function to save the model
    def save_model(model, optimizer, epoch, loss, path):
        if rank == 0:  # Save only on the main process
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, path)
            print(f"Model saved to {path}")

    # Main training loop
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss, train_rouge = train_epoch(ddp_model, train_loader, optimizer, criterion)
        val_loss, val_rouge = validate(ddp_model, val_loader, criterion)

        if rank == 0:  # Print only on the main process
            print(f'Train Loss: {train_loss:.4f}, Train ROUGE: {train_rouge}')
            print(f'Val Loss: {val_loss:.4f}, Val ROUGE: {val_rouge}')

        # Save the model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(ddp_model, optimizer, epoch, val_loss, 'best_model.pth')

        # Save a checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_model(ddp_model, optimizer, epoch, val_loss, f'checkpoint_epoch_{epoch+1}.pth')

    # Save the final model
    save_model(ddp_model, optimizer, EPOCHS, val_loss, 'final_model.pth')

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
