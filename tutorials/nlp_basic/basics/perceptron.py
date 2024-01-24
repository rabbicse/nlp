import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
import numpy as np


class Perceptron(nn.Module):
    """ A perceptron is one linear layer """

    def __init__(self, input_dim):
        """
        Args:
        input_dim (int): size of the input features
        """
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, x_in):
        """The forward pass of the perceptron
        Args:
        x_in (torch.Tensor): an input data tensor
        x_in.shape should be (batch, num_features)
        Returns:
        the resulting tensor. tensor.shape should be (batch,).
        """
        return torch.sigmoid(self.fc1(x_in))


LEFT_CENTER = (3, 3)
RIGHT_CENTER = (3, -2)


def get_toy_data(batch_size, left_center=LEFT_CENTER, right_center=RIGHT_CENTER):
    x_data = []
    y_targets = np.zeros(batch_size)
    for batch_i in range(batch_size):
        if np.random.random() > 0.5:
            x_data.append(np.random.normal(loc=left_center))
        else:
            x_data.append(np.random.normal(loc=right_center))
            y_targets[batch_i] = 1
    return torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_targets, dtype=torch.float32)


def visualize_results(perceptron, x_data, y_truth, n_samples=1000, ax=None, epoch=None,
                      title='', levels=[0.3, 0.4, 0.5], linestyles=['--', '-', '--']):
    y_pred = perceptron(x_data)
    y_pred = (y_pred > 0.5).long().data.numpy().astype(np.int32)

    x_data = x_data.data.numpy()
    y_truth = y_truth.data.numpy().astype(np.int32)

    n_classes = 2

    all_x = [[] for _ in range(n_classes)]
    all_colors = [[] for _ in range(n_classes)]

    colors = ['black', 'white']
    markers = ['o', '*']

    for x_i, y_pred_i, y_true_i in zip(x_data, y_pred, y_truth):
        all_x[y_true_i].append(x_i)
        if y_pred_i == y_true_i:
            all_colors[y_true_i].append("white")
        else:
            all_colors[y_true_i].append("black")
        # all_colors[y_true_i].append(colors[y_pred_i])

    all_x = [np.stack(x_list) for x_list in all_x]

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))

    for x_list, color_list, marker in zip(all_x, all_colors, markers):
        ax.scatter(x_list[:, 0], x_list[:, 1], edgecolor="black", marker=marker, facecolor=color_list, s=300)

    xlim = (min([x_list[:, 0].min() for x_list in all_x]),
            max([x_list[:, 0].max() for x_list in all_x]))

    ylim = (min([x_list[:, 1].min() for x_list in all_x]),
            max([x_list[:, 1].max() for x_list in all_x]))

    # hyperplane

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = perceptron(torch.tensor(xy, dtype=torch.float32)).detach().numpy().reshape(XX.shape)
    ax.contour(XX, YY, Z, colors='k', levels=levels, linestyles=linestyles)

    plt.suptitle(title)

    if epoch is not None:
        plt.text(xlim[0], ylim[1], "Epoch = {}".format(str(epoch)))


def train_perceptron_model():
    input_dim = 2
    lr = 0.001
    batch_size = 1000
    n_epochs = 12
    n_batches = 5

    seed = 1337
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    perceptron = Perceptron(input_dim=input_dim)
    bce_loss = nn.BCELoss()
    optimizer = optim.Adam(params=perceptron.parameters(), lr=lr)

    losses = []

    x_data_static, y_truth_static = get_toy_data(batch_size)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    visualize_results(perceptron, x_data_static, y_truth_static, ax=ax, title='Initial Model State')
    plt.axis('off')

    change = 1.0
    last = 10.0
    epsilon = 1e-3

    # each epoch is a complete pass over the training data
    for epoch_i in range(n_epochs):
        # the inner loop is over the batches in the dataset
        for batch_i in range(n_batches):
            # Step 0: Get the data
            x_data, y_target = get_toy_data(batch_size)
            # Step 1: Clear the gradients
            perceptron.zero_grad()
            # Step 2: Compute the forward pass of the model
            y_pred = perceptron(x_data, apply_sigmoid=True)
            # Step 3: Compute the loss value that we wish to optimize
            loss = bce_loss(y_pred, y_target)
            # Step 4: Propagate the loss signal backward
            loss.backward()
            # Step 5: Trigger the optimizer to perform one update
            optimizer.step()

            loss_value = loss.item()
            losses.append(loss_value)

            change = abs(last - loss_value)
            last = loss_value

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        visualize_results(perceptron, x_data_static, y_truth_static, ax=ax, epoch=epoch_i,
                          title=f"{loss_value}; {change}")
        plt.axis('off')


if __name__ == '__main__':
    train_perceptron_model()
