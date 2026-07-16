import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = os.path.expanduser("~/models/nova-router-v2")

with open(f"{MODEL_PATH}/label_map.json") as f:
    ID2LABEL = {int(k): v for k, v in json.load(f).items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def classify(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    confidence = probs.max().item()
    label = ID2LABEL[logits.argmax(dim=1).item()]
    return label, confidence

# Test with unseen commands
tests = [
    "hey open chrome",
    "volume down please",
    "what is the stock market",
    "play something chill",
    "good night nova",
    "make a new directory",
    "search for arch linux tips",
    "mute the audio",
    "who invented electricity",
    "copy this file to desktop",
    "convert this pdf to docx",
    "change this mp4 to a gif",
]

print(f"{'Input':<40} {'Label':<20} {'Confidence'}")
print("-" * 70)
for t in tests:
    label, conf = classify(t)
    print(f"{t:<40} {label:<20} {conf:.2%}")
