import pandas as pd
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

model = None
tokenizer = None

# ==========================
# CREDIBILITY SCORE
# ==========================

def get_credibility_score(url):
    domain = urlparse(url).netloc.lower()

    if "who.int" in domain:
        return 0.9
    elif ".gov" in domain:
        return 0.8
    elif ".edu" in domain:
        return 0.7
    else:
        return 0.3


# ==========================
# MODEL
# ==========================

class RobertaWithCredibility(nn.Module):

    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.classifier = nn.Linear(769, 2)

    def forward(self, input_ids, attention_mask, credibility):

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = outputs.last_hidden_state[:, 0, :]

        credibility = credibility.float().unsqueeze(1)

        combined = torch.cat((pooled_output, credibility), dim=1)

        logits = self.classifier(combined)

        return logits


# ==========================
# DATASET
# ==========================

class HealthDataset(Dataset):

    def __init__(self, texts, labels, credibility, tokenizer):

        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256
        )

        self.labels = labels
        self.credibility = credibility

    def __getitem__(self, idx):

        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}

        item["labels"] = torch.tensor(self.labels[idx])
        item["credibility"] = torch.tensor(self.credibility[idx], dtype=torch.float)

        return item

    def __len__(self):
        return len(self.labels)


# ==========================
# TRAIN FUNCTION
# ==========================

def train_model():

    global model, tokenizer

    print("\nLoading dataset...")

    fake_df = pd.read_csv("data/NewsFakeCOVID-19.csv")[["content", "news_url"]].dropna()
    real_df = pd.read_csv("data/NewsRealCOVID-19.csv")[["content", "news_url"]].dropna()

    fake_df["label"] = 1
    real_df["label"] = 0

    real_df = real_df.sample(n=len(fake_df), random_state=42)

    df = pd.concat([fake_df, real_df]).sample(frac=1).reset_index(drop=True)

    df["credibility"] = df["news_url"].apply(get_credibility_score)

    train_texts, test_texts, train_labels, test_labels, train_cred, test_cred = train_test_split(
        df["content"].tolist(),
        df["label"].tolist(),
        df["credibility"].tolist(),
        test_size=0.2,
        random_state=42
    )

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    train_dataset = HealthDataset(train_texts, train_labels, train_cred, tokenizer)
    test_dataset = HealthDataset(test_texts, test_labels, test_cred, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    model = RobertaWithCredibility()

    optimizer = AdamW(model.parameters(), lr=5e-5)

    loss_fn = nn.CrossEntropyLoss()

    print("\nTraining started...\n")

    model.train()

    for epoch in range(3):

        total_loss = 0

        for batch in train_loader:

            optimizer.zero_grad()

            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                credibility=batch["credibility"]
            )

            loss = loss_fn(logits, batch["labels"])

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss}")

    print("\nEvaluating model...\n")

    model.eval()

    preds = []
    labels = []
    probs = []

    with torch.no_grad():

        for batch in test_loader:

            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                credibility=batch["credibility"]
            )

            probabilities = F.softmax(logits, dim=1)

            p = torch.argmax(probabilities, dim=1)

            preds.extend(p.tolist())
            labels.extend(batch["labels"].tolist())
            probs.extend(probabilities[:,1].tolist())

    accuracy = accuracy_score(labels, preds)

    print("\nTest Accuracy:", accuracy)

    print("\nClassification Report:\n")

    print(classification_report(labels, preds))


    # ==========================
    # CONFUSION MATRIX
    # ==========================

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"]
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Confusion Matrix - Health Misinformation Detection")

    plt.show()


    # ==========================
    # ROC CURVE
    # ==========================

    fpr, tpr, thresholds = roc_curve(labels, probs)

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))

    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")

    plt.plot([0,1], [0,1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Health Misinformation Detection")

    plt.legend()

    plt.show()

    print("\n✅ Training completed. Graphs generated.\n")


# ==========================
# DETECT NEWS
# ==========================

def detect_news():

    global model, tokenizer

    if model is None:
        print("\n⚠ Train the model first (Option 1)\n")
        return

    while True:

        text = input("\nEnter article text (type EXIT to stop):\n")

        if text.lower() == "exit":
            break

        url = input("Enter source URL:\n")

        cred = get_credibility_score(url)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        with torch.no_grad():

            logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                credibility=torch.tensor([cred], dtype=torch.float)
            )

        prediction = torch.argmax(logits, dim=1).item()

        if prediction == 1:
            print("\nPrediction: FAKE NEWS\n")
        else:
            print("\nPrediction: REAL NEWS\n")


# ==========================
# MAIN MENU LOOP
# ==========================

while True:

    print("\n===== HEALTH MISINFORMATION DETECTOR =====")
    print("1. Train Model")
    print("2. Detect Fake News")
    print("3. Exit")

    choice = input("\nEnter choice: ")

    if choice == "1":
        train_model()

    elif choice == "2":
        detect_news()

    elif choice == "3":
        print("\nExiting program...\n")
        break

    else:
        print("\nInvalid option\n")