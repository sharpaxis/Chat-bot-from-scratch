import json
import re
import numpy as np
from nltk_utils import tokenize,bag_of_words
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from model import NeuralNet 

with open('intents.json','r') as f:
    intents = json.load(f)
all_words = [] #this will contain all words in the intents file
tags = []       #this will contain actual intent tag
xy = []         #both tags and sentence in label and data format
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    # patterns are individual sentences, we apply tokenization to each sentence and get all_words in token format
    for pattern in intent['patterns']:
        w = tokenize(pattern)# this will return an list which we will extend to all words
        all_words.extend(w)
        #in xy we append both the pattern and the intent in a tuple(w,tag)
        xy.append((w,tag))
#we're done tokenizing its time to lower and stem the words
text = ' '.join(all_words)
clean = ' '.join(re.findall(r"[^?!.,/]+",text.lower()))
all_words = clean.split()
#only take unique words
all_words = sorted(set(all_words))
tags = sorted(set(tags))
X_train = []
y_train = []
for (pattern_setence,tag) in xy:
    bag = bag_of_words(pattern_setence,all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)# 1 hot encoding
X_train = np.array(X_train)
y_train = np.array(y_train)

#dataset
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    #dataset index
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.n_samples

#hyper parameters
batch_size = 8 
learning_rate = 0.001
num_epochs = 1000
#defining sizes
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0]) #this gives size of each bow vector which will be fed into initial layer   

dataset = ChatDataset()

train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size,hidden_size,output_size).to(device=device) 
#loss and optimizer

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

#training
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device) 
        labels = labels.to(device)

        #forward
        words = words.float()
        outputs = model(words) 
        loss = criterion(outputs,labels)

        #backward and optimizer step
        #review
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
         
    if (epoch+1)%100 ==0:
        print(f'epoch {epoch+1}/{num_epochs}, loss = {loss.item():.4f}')
print(f'final loss,  loss = {loss.item():.4f}') 

#saving the model
data = {
    "model_state":model.state_dict(),
    "input_size": input_size,
    "hidden_size":hidden_size,
    "output_size":output_size,
    "all_words":all_words,
    "tags":tags,

}
FILE = "data.pth"
torch.save(data,FILE)

print(f'training complete. file saved to {FILE}')
