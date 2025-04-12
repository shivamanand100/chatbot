from model import ChatbotNN
import json, torch, torch.nn as nn, torch.optim as optim, numpy as np, nltk, pickle
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("wordnet")

with open("intents.json", "r", encoding="utf-8") as file:
    intents_data = json.load(file)

lemmatizer = WordNetLemmatizer()

def tokenize_and_lemmatize(sentence):
    words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in words]

def bag_of_words(tokenized_sentence, words):
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in tokenized_sentence:
            bag[idx] = 1
    return bag

all_words, tags, patterns = [], [], []

for intent in intents_data["intents"]:
    for pattern in intent["patterns"]:
        word_list = tokenize_and_lemmatize(pattern)
        all_words.extend(word_list)
        patterns.append((word_list, intent["tag"]))
    tags.append(intent["tag"])

all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train, y_train = [], []

for (pattern_sentence, tag) in patterns:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    y_train.append(tags.index(tag))

X_train = np.array(X_train)
y_train = np.array(y_train)

input_size, hidden_size, output_size = len(all_words), 8, len(tags)
model = ChatbotNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(2000):
    optimizer.zero_grad()
    outputs = model(torch.tensor(X_train, dtype=torch.float32))
    loss = criterion(outputs, torch.tensor(y_train, dtype=torch.long))
    loss.backward()
    optimizer.step()

data = {"model": model, "all_words": all_words, "tags": tags, "intents_data": intents_data}
with open("model.pkl", "wb") as f:
    pickle.dump(data, f)

print("Training complete! Model saved to model.pkl")
import torch
import numpy as np
with torch.no_grad():
    outputs = model(torch.tensor(X_train, dtype=torch.float32))
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == torch.tensor(y_train, dtype=torch.long)).sum().item() / len(y_train)
    print(f"Accuracy of the model: {accuracy}")