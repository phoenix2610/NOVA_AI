import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.expanduser("~/models/deberta-v3-base")
DATASET_PATH = os.path.expanduser("~/nova/router/dataset.json")
SAVE_PATH    = os.path.expanduser("~/models/nova-router-v2")

# ── Labels — 7 classes ────────────────────────────────────────────────────────
LABELS   = ["browser_op", "system_op", "media_op", "knowledge_query",
            "conversation", "file_op", "conversion_op"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

# ── conversion_op examples to append ─────────────────────────────────────────
CONVERSION_EXAMPLES = [
    # Documents
    {"input": "convert this docx to pdf",              "label": "conversion_op"},
    {"input": "turn my word document into a pdf",      "label": "conversion_op"},
    {"input": "export the file as pdf",                "label": "conversion_op"},
    {"input": "convert pdf to word",                   "label": "conversion_op"},
    {"input": "change this document to epub",          "label": "conversion_op"},
    {"input": "convert my markdown to html",           "label": "conversion_op"},
    {"input": "turn this html into a pdf",             "label": "conversion_op"},
    {"input": "convert odt to docx",                   "label": "conversion_op"},
    {"input": "make this a pdf file",                  "label": "conversion_op"},
    {"input": "convert the document to markdown",      "label": "conversion_op"},
    # Images
    {"input": "convert png to jpg",                    "label": "conversion_op"},
    {"input": "change this image to webp",             "label": "conversion_op"},
    {"input": "convert the photo to jpeg",             "label": "conversion_op"},
    {"input": "convert heic to png",                   "label": "conversion_op"},
    {"input": "change image format to tiff",           "label": "conversion_op"},
    {"input": "convert this picture to jpg",           "label": "conversion_op"},
    {"input": "export image as webp",                  "label": "conversion_op"},
    {"input": "turn this into a png file",             "label": "conversion_op"},
    # Audio
    {"input": "convert mp3 to wav",                    "label": "conversion_op"},
    {"input": "change audio format to flac",           "label": "conversion_op"},
    {"input": "convert this song to opus",             "label": "conversion_op"},
    {"input": "turn wav into mp3",                     "label": "conversion_op"},
    {"input": "convert aac to mp3",                    "label": "conversion_op"},
    {"input": "export audio as flac",                  "label": "conversion_op"},
    {"input": "change this to a wav file",             "label": "conversion_op"},
    {"input": "convert audio file to aac",             "label": "conversion_op"},
    # Video
    {"input": "convert mp4 to mkv",                    "label": "conversion_op"},
    {"input": "change video format to avi",            "label": "conversion_op"},
    {"input": "convert this video to h265",            "label": "conversion_op"},
    {"input": "turn mkv into mp4",                     "label": "conversion_op"},
    {"input": "convert avi to mp4",                    "label": "conversion_op"},
    {"input": "export video as mkv",                   "label": "conversion_op"},
    {"input": "convert video to av1",                  "label": "conversion_op"},
    {"input": "turn this into an mp4",                 "label": "conversion_op"},
    # Data
    {"input": "convert csv to json",                   "label": "conversion_op"},
    {"input": "change this json to csv",               "label": "conversion_op"},
    {"input": "convert excel to csv",                  "label": "conversion_op"},
    {"input": "turn csv into xlsx",                    "label": "conversion_op"},
    {"input": "convert json to xml",                   "label": "conversion_op"},
    {"input": "export data as parquet",                "label": "conversion_op"},
    {"input": "convert xml to csv",                    "label": "conversion_op"},
    {"input": "turn this spreadsheet into json",       "label": "conversion_op"},
    {"input": "convert xlsx to parquet",               "label": "conversion_op"},
    # Ebooks
    {"input": "convert epub to mobi",                  "label": "conversion_op"},
    {"input": "change ebook format to pdf",            "label": "conversion_op"},
    {"input": "convert mobi to epub",                  "label": "conversion_op"},
    {"input": "turn this epub into azw3",              "label": "conversion_op"},
    {"input": "convert pdf to epub",                   "label": "conversion_op"},
    {"input": "hey can you convert this file for me",  "label": "conversion_op"},
    {"input": "change the format of this document",    "label": "conversion_op"},
    {"input": "transform this video file to mp4",      "label": "conversion_op"},
    {"input": "export this as a different format",     "label": "conversion_op"},
    {"input": "convert the recording to flac",         "label": "conversion_op"},
    {"input": "turn the image into a jpeg",            "label": "conversion_op"},
    {"input": "make this an epub please",              "label": "conversion_op"},
    {"input": "convert my notes to pdf",               "label": "conversion_op"},
    {"input": "change this mp4 to a gif",              "label": "conversion_op"},
]

# ── Dataset class ─────────────────────────────────────────────────────────────
class RouterDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=64):
        self.data      = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item     = self.data[idx]
        encoding = self.tokenizer(
            item["input"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label":          torch.tensor(LABEL2ID[item["label"]], dtype=torch.long)
        }

# ── Load and extend dataset ───────────────────────────────────────────────────
print("Loading dataset...")
with open(DATASET_PATH) as f:
    data = json.load(f)

# Remove any existing conversion_op entries to avoid duplication on re-run
data = [d for d in data if d["label"] != "conversion_op"]
data.extend(CONVERSION_EXAMPLES)

MORE_EXAMPLES = [
    {"input": "hey open chrome", "label": "browser_op"},
    {"input": "open chrome browser", "label": "browser_op"},
    {"input": "launch google chrome", "label": "browser_op"},
    {"input": "mute the audio", "label": "system_op"},
    {"input": "mute the sound", "label": "system_op"},
    {"input": "mute audio", "label": "system_op"},
]
data.extend(MORE_EXAMPLES)

from collections import Counter
dist = Counter(d["label"] for d in data)
print(f"Total examples: {len(data)}")
for k, v in sorted(dist.items()):
    print(f"  {k:<22} {v}")

# Save updated dataset
updated_path = os.path.expanduser("~/nova/router/dataset_v2.json")
with open(updated_path, "w") as f:
    json.dump(data, f, indent=2)
print(f"Updated dataset saved → {updated_path}")

# ── Load tokenizer ────────────────────────────────────────────────────────────
print("Loading DeBERTa-v3-large tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Load model with classification head ──────────────────────────────────────
print("Loading model with 7-class classification head...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=len(LABELS),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    local_files_only=True,
    ignore_mismatched_sizes=True,
    torch_dtype=torch.bfloat16,
)

# Change back to:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")

# ── Dataloader ────────────────────────────────────────────────────────────────
import random
random.seed(42)
random.shuffle(data)

split      = int(len(data) * 0.85)
train_data = data[:split]
eval_data  = data[split:]

train_loader = DataLoader(RouterDataset(train_data, tokenizer), batch_size=1,  shuffle=True)
eval_loader  = DataLoader(RouterDataset(eval_data,  tokenizer), batch_size=4, shuffle=False)

print(f"Train: {len(train_data)} | Eval: {len(eval_data)}")

# ── Optimizer + scheduler ─────────────────────────────────────────────────────
EPOCHS = 40
optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Training loop ─────────────────────────────────────────────────────────────
print("\nTraining started...")
best_eval_acc = 0.0

for epoch in range(EPOCHS):
    # Train
    model.train()
    total_loss = 0
    correct    = 0
    optimizer.zero_grad()
    for i, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / 8
            logits = outputs.logits

        loss.backward()
        
        if (i + 1) % 8 == 0 or (i + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * 8
        correct    += (logits.argmax(dim=1) == labels).sum().item()

    train_acc = correct / len(train_data) * 100
    scheduler.step()

    # Eval
    model.eval()
    eval_correct = 0
    with torch.no_grad():
        for batch in eval_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            eval_correct  += (outputs.logits.argmax(dim=1) == labels).sum().item()

    eval_acc = eval_correct / len(eval_data) * 100
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f} — "
          f"Train: {train_acc:.1f}% — Eval: {eval_acc:.1f}%")

    # Save best model
    if eval_acc > best_eval_acc:
        best_eval_acc = eval_acc
        os.makedirs(SAVE_PATH, exist_ok=True)
        model.save_pretrained(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)
        with open(f"{SAVE_PATH}/label_map.json", "w") as f:
            json.dump({str(i): l for i, l in ID2LABEL.items()}, f, indent=2)
        print(f"  ✓ Best model saved (eval: {eval_acc:.1f}%)")

print(f"\nTraining complete. Best eval accuracy: {best_eval_acc:.1f}%")
print(f"Router saved → {SAVE_PATH}")
