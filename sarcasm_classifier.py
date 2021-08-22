import json

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000
num_epochs = 32

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.net = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim, max_norm = max_length),
            nn.AvgPool1d(embedding_dim),
            nn.Flatten(1),
            nn.Linear(max_length, 24),
            nn.ReLU(),
            nn.Linear(24, 1)
        )

    def forward(self, x):
        return self.net(x)

class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.sequences = []
        self.labels = []

        self.load_dataset()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.labels[idx]).unsqueeze(-1)

    def load_dataset(self):
        with open("dataset/sarcasm.json", 'r') as f:
            datastore = json.load(f)

        sentences = []
        labels = []

        for item in datastore:
            sentences.append(item['headline'])
            labels.append(item['is_sarcastic'])

        tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
        tokenizer.fit_on_texts(sentences)

        sequences = tokenizer.texts_to_sequences(sentences)
        sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

        self.sequences = sequences
        self.labels = labels

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net().to(device)
bceloss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr = 0.001)
scaler = torch.cuda.amp.GradScaler()

dataset = SarcasmDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle = True, num_workers = 8)

for epoch in range(num_epochs):
    total_loss = 0
    for i, data in enumerate(dataloader, 0):
        sequences, labels = data
        sequences, labels = sequences.to(device), labels.float().to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = net(sequences)
            loss = bceloss(output, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    print('Avg Loss: %.3f' % (total_loss / i))

print('Finished Training')
