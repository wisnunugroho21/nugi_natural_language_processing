import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn
from torch.nn.functional import softmax

vocab_size = 10000
embedding_dim = 100
max_length = 15
trunc_type='pre'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000
num_epochs = 100

class SingleLengthSelfAttention(nn.Module):
    def __init__(self, dim, dim_i):
        super(SingleLengthSelfAttention, self).__init__()

        self.scaling_factor = torch.tensor(dim).sqrt()

        self.net_value = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
        )

        self.net_key = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
        )

        self.net_query = nn.Sequential(
            nn.Linear(dim * dim_i, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
        )

        self.net_output = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
        )

    def forward(self, x):
        value   = self.net_value(x)
        key     = self.net_key(x)
        query   = self.net_query(x.flatten(1)).unsqueeze(1)

        attn_scores         = query @ key.transpose(1, 2)
        scaled_attn_scores  = attn_scores / self.scaling_factor
        attn_scores_softmax = softmax(scaled_attn_scores, dim = -1)
        weighted_value      = attn_scores_softmax @ value

        outputs = self.net_output(weighted_value.squeeze())
        return outputs

class MultiLengthSelfAttention(nn.Module):
    def __init__(self, dim):
        super(MultiLengthSelfAttention, self).__init__()

        self.scaling_factor = torch.tensor(dim).sqrt()

        self.net_value = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
        )

        self.net_key = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
        )

        self.net_query = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
        )

        self.net_output = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
        )

    def forward(self, x):
        value   = self.net_value(x)
        key     = self.net_key(x)
        query   = self.net_query(x)

        attn_scores         = query @ key.transpose(1, 2)
        scaled_attn_scores  = attn_scores / self.scaling_factor
        attn_scores_softmax = softmax(scaled_attn_scores, dim = -1)
        weighted_value      = attn_scores_softmax @ value

        outputs = self.net_output(weighted_value.squeeze())
        return outputs

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.net = nn.Sequential(
            nn.Embedding(2690, embedding_dim, max_norm = 15),
            SingleLengthSelfAttention(embedding_dim, 15),
            nn.Linear(embedding_dim, 150),
            nn.GELU(),
            nn.Linear(150, 2690)
        )

    def forward(self, x):
        return self.net(x)

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
        input_sequences = np.array(pad_sequences(input_sequences, maxlen = max_sequence_len, padding = padding_type))

        # create predictors and label
        sequences, labels = input_sequences[:,:-1],input_sequences[:,-1]

        self.sequences = sequences
        self.labels = labels

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net         = Net().to(device)
bceloss     = nn.CrossEntropyLoss()
optimizer   = torch.optim.AdamW(net.parameters(), lr = 0.001)
scaler      = torch.cuda.amp.GradScaler()

dataset     = LyricsDataset()
dataloader  = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle = True, num_workers = 8)

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

    print('{}/{} Avg Loss: {:.3f}'.format(i, epoch, (total_loss / i)))

print('Finished Training')
