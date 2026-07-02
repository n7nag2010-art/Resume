# ✅ IMPORTS
import pandas as pd
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ✅ STEP 1: LOAD EXCEL

df = pd.read_excel(r'E:/nagarajn/Projects/Agentic/PyTorch/payer_data.xlsx', engine="openpyxl")


print("✅ Original Data Shape:", df.shape)

# ✅  Check Null Values
print("\n✅ NULL VALUE CHECK:")
print(df.isnull().sum())

# ✅  Drop Null rows (important)
df = df.dropna(subset=["EOB Payer Name", "Common PayerName"])

# ✅  Remove Duplicate Rows
df = df.drop_duplicates()

print("\n✅ After Removing Null & Duplicates:", df.shape)


# ✅ STEP 2: CLEAN TEXT
def clean_text(text):
    text = str(text).lower()
    text = text.strip()
    text = re.sub(r'[^a-z0-9 ]', '', text)  # remove special chars
    text = re.sub(r'\s+', ' ', text)        # remove extra spaces
    return text

df["EOB Payer Name"] = df["EOB Payer Name"].apply(clean_text)

# ✅ STEP 3: TOKENIZATION
def tokenize(text):
    return text.split()

df["tokens"] = df["EOB Payer Name"].apply(tokenize)

# ✅ STEP 4: BUILD VOCABULARY (WORD → ID)
vocab = {"<PAD>": 0, "<UNK>": 1}

for tokens in df["tokens"]:
    for word in tokens:
        if word not in vocab:
            vocab[word] = len(vocab)
print(vocab)
print(len(vocab))
# ✅ STEP 5: CONVERT TOKENS → IDS
max_len = 10

def encode(tokens):
    ids = [vocab.get(word, vocab["<UNK>"]) for word in tokens]
    
    # padding
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    
    return ids

df["input_ids"] = df["tokens"].apply(encode)

# ✅ STEP 6: LABEL ENCODING
labels = df["Common PayerName"].unique()
label_map = {label: i for i, label in enumerate(labels)}
df["label"] = df["Common PayerName"].map(label_map)

# ✅ STEP 7: TRAIN TEST SPLIT
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# ✅ DATASET CLASS
class PayerDataset(Dataset):
    def __init__(self, data):
        self.X = list(data["input_ids"])
        self.y = list(data["label"])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.long),
        )

train_data = PayerDataset(train_df)
test_data = PayerDataset(test_df)

train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
test_loader = DataLoader(test_data, batch_size=4)

# ✅ STEP 8: MODEL (Embedding + Neural Network)
class PayerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(PayerModel, self).__init__()
        print(vocab_size)
        print(embed_dim)
        print(num_classes)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim * max_len, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)


    def forward(self, x):
        x = self.embedding(x)
        #print(x)
        x = x.view(x.size(0), -1)
        #print(x)
        x = self.relu(self.fc1(x))
        #print(x)
        x = self.fc2(x)
        #print(x)
        return x
        
 

model = PayerModel(
    vocab_size=len(vocab),
    embed_dim=50,
    num_classes=len(label_map)
)

# ✅ STEP 9: LOSS + OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ✅ STEP 10: TRAINING LOOP (CORE PART YOU ASKED)
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:

        # ✅ FORWARD PASS
        outputs = model(X_batch)

        # ✅ LOSS
        loss = criterion(outputs, y_batch)

        # ✅ BACKWARD PASS
        optimizer.zero_grad()
        loss.backward()

        # ✅ UPDATE WEIGHTS
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ✅ STEP 11: TEST ACCURACY
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)

        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

print("✅ Accuracy:", correct / total)

# ✅ STEP 12: PREDICTION FUNCTION
def predict(text):
    text = clean_text(text)
    tokens = tokenize(text)
    ids = encode(tokens)
    print(ids)
    print(torch.long)
    tensor = torch.tensor([ids], dtype=torch.long)
    print(tensor.shape)

    with torch.no_grad():
        outputs = model(tensor)
        _, pred = torch.max(outputs, 1)

    reverse_map = {v: k for k, v in label_map.items()}
    return reverse_map[pred.item()]

# ✅ TEST
print("\nPrediction:", predict("EXCELLUS BLUECROSS BLUESHIELD"))