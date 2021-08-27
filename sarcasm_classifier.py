import json

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn
from torch.nn.functional import softmax

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 25000
num_epochs = 100

class MaskedSoftmax(nn.Module):
    def __init__(self):
        super(MaskedSoftmax, self).__init__()

    def forward(self, input: torch.Tensor, bool_mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
        min_type_value  = torch.finfo(input.dtype).min
        masked_value    = input.masked_fill(bool_mask, min_type_value)

        return softmax(masked_value, dim = dim)

class SelfAttention(nn.Module):
    def __init__(self, num_dim):
        super(SelfAttention, self).__init__()

        self.scaling_factor = torch.tensor(num_dim).sqrt()

    def forward(self, value, key, query):
        attn_scores         = query @ key.transpose(1, 2)
        scaled_attn_scores  = attn_scores / self.scaling_factor
        attn_scores_softmax = softmax(scaled_attn_scores, dim = -1)
        outputs             = attn_scores_softmax @ value
        
        return outputs

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
        )

        self.net_output = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
        )

    def forward(self, x):
        value   = self.net_value(x)
        key     = self.net_key(x)
        query   = self.net_query(x.flatten(1).unsqueeze(1))

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
            nn.Embedding(vocab_size, embedding_dim, max_norm = max_length),
            SingleLengthSelfAttention(embedding_dim, max_length),
            nn.Linear(embedding_dim, 24),
            nn.GELU(),
            nn.Linear(24, 1)
        )

    def forward(self, x):
        return self.net(x)

class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, is_training, training_size = 20000):
        self.is_training = is_training
        self.training_size = training_size

        self.sequences  = []
        self.labels     = []

        self.load_dataset()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.labels[idx]).unsqueeze(-1)

    def load_dataset(self):
        with open("dataset/sarcasm.json", 'r') as f:
            datastore = json.load(f)

        sentences   = []
        labels      = []

        for item in datastore:
            sentences.append(item['headline'])
            labels.append(item['is_sarcastic'])

        tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
        tokenizer.fit_on_texts(sentences)

        sequences = tokenizer.texts_to_sequences(sentences)
        sequences = pad_sequences(sequences, maxlen = max_length, padding = padding_type, truncating = trunc_type)

        if self.is_training:
            self.sequences  = sequences[:self.training_size]
            self.labels     = labels[:self.training_size]
        else: 
            self.sequences  = sequences[self.training_size:]
            self.labels     = labels[self.training_size:]

        print('Finished Load Data')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net         = Net().to(device)
bceloss     = nn.BCEWithLogitsLoss()
optimizer   = torch.optim.AdamW(net.parameters(), lr = 0.001)
scaler      = torch.cuda.amp.GradScaler()

traindataset = SarcasmDataset(True, training_size)
trainloader = torch.utils.data.DataLoader(traindataset, batch_size = 32, shuffle = True, num_workers = 8)

testdataset = SarcasmDataset(False, training_size)
testloader  = torch.utils.data.DataLoader(testdataset, batch_size = 32, shuffle = False, num_workers = 8)

for epoch in range(num_epochs):
    total_loss = 0
    for i, data in enumerate(trainloader, 0):
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

    average_loss = total_loss / i
    print('Epoch: {} | Avg Loss: {:.3f}'.format(epoch, average_loss))

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        sequences, labels = data
        sequences, labels = sequences.to(device), labels.float().to(device)

        output = net(sequences)
        output = torch.where(
            output >= 0.5, 
            torch.tensor(1).to(device), 
            torch.tensor(0).to(device)
        )

        total   += labels.size(0)
        correct += (output == labels).sum().item()

correct_percentage = 100 * correct / total
print("Accuracy on test dataset: {:.1f} %".format(correct_percentage))
