import numpy as np

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
num_epochs = 100

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.net = nn.Embedding(2690, 100, max_norm = max_length)

        self.lstm = nn.LSTM(100, 150, batch_first = True)
        self.out = nn.Linear(150, 2690)

    def forward(self, x):
        x   = self.net(x)     
        x, (hn, cn) = self.lstm(x)
        return self.out(x[:, -1])

class LyricsDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.sequences = []
        self.labels = []

        self.load_dataset()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.labels[idx])

    def load_dataset(self):
        tokenizer = Tokenizer()

        data = open('dataset/irish-lyrics-eof.txt').read()
        corpus = data.lower().split("\n")

        tokenizer.fit_on_texts(corpus)

        input_sequences = []
        for line in corpus:
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[: i+1 ]
                input_sequences.append(n_gram_sequence)

        # pad sequences 
        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen = max_sequence_len, padding='pre'))

        # create predictors and label
        sequences, labels = input_sequences[:,:-1],input_sequences[:,-1]

        self.sequences = sequences
        self.labels = labels

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net().to(device)
bceloss = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr = 0.001)
scaler = torch.cuda.amp.GradScaler()

dataset = LyricsDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle = True, num_workers = 8)

for epoch in range(num_epochs):
    total_loss = 0
    for i, data in enumerate(dataloader, 0):
        sequences, labels = data
        sequences, labels = sequences.to(device), labels.long().to(device)

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
