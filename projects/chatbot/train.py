import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.neural_net import NeuralNet
from utils.nltk_utils import bag_of_words, tokenize, stem


class ChatDataset(Dataset):
    def __init__(self, x_train, y_train):
        """
        :param x_train:
        :param y_train:
        """
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


class TrainChatbot:
    def __init__(self, x_train, y_train, tags, all_words, num_epochs: int = 1000, batch_size: int = 8,
                 learning_rate: int = 0.001,
                 hidden_size: int = 8):
        """
        :param num_epochs:
        :param batch_size:
        :param learning_rate:
        :param hidden_size:
        """
        # Hyper-parameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_size = len(x_train[0])
        self.hidden_size = hidden_size
        self.output_size = len(tags)
        self.tags = tags
        self.all_words = all_words

        print(self.input_size, self.output_size)

        self.dataset = ChatDataset(x_train=x_train, y_train=y_train)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NeuralNet(self.input_size, self.hidden_size, self.output_size).to(self.device)

    def train(self):
        """
        Train neural network
        :return:
        """
        train_loader = DataLoader(dataset=self.dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=0)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Train the model
        for epoch in range(self.num_epochs):
            for (words, labels) in train_loader:
                words = words.to(self.device)
                labels = labels.to(dtype=torch.long).to(self.device)

                # Forward pass
                outputs = self.model(words)
                # if y would be one-hot, we must apply
                # labels = torch.max(labels, 1)[1]
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}')

        print(f'final loss: {loss.item():.4f}')

        data = {
            "model_state": self.model.state_dict(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "all_words": self.all_words,
            "tags": self.tags
        }

        output_file = "ml_output/data.pth"
        torch.save(data, output_file)

        print(f'training complete. file saved to {output_file}')


def get_intents():
    with open('data/intents.json', 'r') as f:
        return json.load(f)


def preprocess_data(intents):
    all_words = []
    tags = []
    xy = []
    # loop through each sentence in our intents patterns
    for intent in intents['intents']:
        tag = intent['tag']
        # add to tag list
        tags.append(tag)
        for pattern in intent['patterns']:
            # tokenize each word in the sentence
            w = tokenize(pattern)
            # add to our words list
            all_words.extend(w)
            # add to xy pair
            xy.append((w, tag))

    # stem and lower each word
    ignore_words = ['?', '.', '!']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    # remove duplicates and sort
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    print(len(xy), "patterns")
    print(len(tags), "tags:", tags)
    print(len(all_words), "unique stemmed words:", all_words)

    return xy, tags, all_words


def create_training_data(xy, tags, all_words):
    # create training data
    x_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
        # x: bag of words for each pattern_sentence
        bag = bag_of_words(pattern_sentence, all_words)
        x_train.append(bag)
        # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
        label = tags.index(tag)
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train, y_train


def start_train_chatbot():
    intents = get_intents()
    xy, tags, all_words = preprocess_data(intents=intents)
    x_train, y_train = create_training_data(xy=xy, tags=tags, all_words=all_words)
    train_chatbot = TrainChatbot(x_train=x_train, y_train=y_train, tags=tags, all_words=all_words)
    train_chatbot.train()


if __name__ == '__main__':
    start_train_chatbot()
