import argparse
import sys
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from ascii_graph import Pyasciigraph as pygraph
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader, Subset
from torch_kmeans import KMeans
from torchvision import datasets, transforms

import wandb
from params import Parameters
from topological_maps import LossEfficacyFactory, TopologicalMap


def plot_loss(losses, num_classes=5):
    class_width = len(losses) // num_classes
    if class_width < 1:
        losses = np.tile(losses, [num_classes, 1]).T.flatten()
        class_width = len(losses) // num_classes

    classes = [
        sum(losses[i * class_width : (i + 1) * class_width]) / class_width
        for i in range(num_classes)
    ]
    graph_maker = pygraph(10, 6, float_format="{:5.3f}")
    data = [("", x) for x in classes]
    graph = graph_maker.graph("", data)
    print()
    for line in graph[2:]:
        print(line)
    print()

    return np.array(classes)


def create_data_loaders(train_data, test_data, task, batch_size, subset_size):
    """Creates data loaders for training and testing datasets.

    Args:
        train_data: Training dataset.
        test_data: Testing dataset.
        task: List of target labels to include in the subset.
        batch_size: Batch size for data loaders.
        subset_size: Number of samples to include in the subset.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and testing data loaders.
    """
    # Filter training data based on the specified task
    select = [item in task for item in train_data.targets]
    indices = torch.arange(len(train_data.targets))[select]
    train_subset = Subset(train_data, indices)

    # Filter testing data based on the specified task
    select = [item in task for item in test_data.targets]
    indices = torch.arange(len(test_data.targets))[select]
    test_subset = Subset(test_data, indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def initialize_models(input_dim, latent_dim, learning_rate):
    """Initializes the Topological Map model and optimizer.

    Args:
        input_dim: Dimension of the input data.
        latent_dim: Dimension of the latent space.
        learning_rate: Learning rate for the optimizer.

    Returns:
        Tuple[TopologicalMap, optim.Adam]: Initialized model and optimizer.
    """
    model = TopologicalMap(input_dim, latent_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer


def train_stm(
    model,
    optimizer,
    train_loader,
    loss_manager,
    anchors,
    epochs,
    neigh_sigma_base,
    neigh_sigma_max,
    lr_base,
    lr_max,
    anchor_sigma,
    DEVICE,
):
    """Trains the Spatial-Temporal Map model.

    Args:
        model: The Spatial-Temporal Map model.
        optimizer: The optimizer for training.
        train_loader: The data loader for the training data.
        loss_manager: Manages the loss computation.
        anchors: Anchor points for the loss calculation.
        epochs: The number of training epochs.
        neigh_sigma_base: Base neighborhood sigma value.
        neigh_sigma_max: Maximum neighborhood sigma value.
        lr_base: Base learning rate modulation value.
        lr_max: Maximum learning rate modulation value.
        anchor_sigma: Standard deviation for anchor neighborhoods.
        DEVICE: The device to run the training on (CPU or GPU).
    """
    for epoch in tqdm.tqdm(range(epochs)):
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)

            # Compute loss based on model output and parameters
            loss = loss_manager.loss(
                output,
                neighborhood_baseline=neigh_sigma_base,
                neighborhood_max=neigh_sigma_max,
                modulation_baseline=lr_base,
                modulation_max=lr_max,
                anchors=anchors[target],
                neighborhood_std_anchors=anchor_sigma,
            )

            loss.backward()
            optimizer.step()


def to_numpy(x):
    """Converts a PyTorch tensor to a NumPy array.

    Args:
        x: The PyTorch tensor.

    Returns:
        numpy.ndarray: The NumPy array.
    """
    return x.cpu().detach().numpy()



class MultinomialLogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes):
        super(MultinomialLogisticRegression, self).__init__()
        self.linear = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.linear(x)


class MultinomialLogisticRegressionCallable:
    def __init__(
        self,
        learning_rate=0.03,
        num_features=None,
        num_classes=None,
        device="cpu",
    ):
        self.learning_rate = learning_rate
        self.num_features = num_features
        self.num_classes = num_classes
        self.model = None
        self.optimizer = None
        self.device = device

        self.model = MultinomialLogisticRegression(
            self.num_features, self.num_classes
        ).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

    def __call__(self, X, y):
        if self.num_features is None:
            self.num_features = X.shape[1]
        if self.num_classes is None:
            self.num_classes = len(set(y))
        y = (
            torch.nn.functional.one_hot(
                torch.tensor(y).long(), num_classes=self.num_classes
            )
            .float()
            .to(self.device)
        )

        self.optimizer.zero_grad()
        outputs = self.model(X)
        loss = self.criterion(outputs, y)
        loss.backward(retain_graph=True)
        self.optimizer.step()

        return loss.item()


def get_regression(
    model,
    train_loaders,
    device,
    num_classes=10,
    num_iterations=200,
):
    num_features = model.output_size
    regressionManager = MultinomialLogisticRegressionCallable(
        learning_rate=0.03,
        num_features=num_features,
        num_classes=num_classes,
        device=device,
    )

    for epoch in range(num_iterations):
        epoch_loss = []
        for loader in train_loaders:
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                norms = model(data)
                X = nn.functional.softmax(1e3 / norms, 1)
                y = target

                loss = regressionManager(X, y)
                epoch_loss.append(loss)
        print(
            f"Regress on task - epoch: {epoch:>4d}/{num_iterations} -- "
            f"loss: {np.mean(epoch_loss):5.3f}"
        )

    return regressionManager.model


def evaluate_model_by_regression(model, test_loaders, regress_model, device):

    accuracies = []
    for loader in test_loaders:
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            norms = model(data)
            X = nn.functional.softmax(1e3 / norms, 1)
            y = target
            predict_hovs = regress_model(X)
            predict = predict_hovs.argmax(1)
            accuracies.append(
                (y == predict).sum() / len(y),
            )
    accurate = torch.tensor(accuracies).mean().item()
    print(f"accuracy is {100*accurate:<5.2f}%")
    return accurate



class Plotter:
    """A class for plotting weights and efficacies during training."""

    def __init__(self, model, loss_manager, side, imside, cmap):
        """Initializes the Plotter.

        Args:
            model: The model to plot weights from.
            loss_manager: The loss manager to plot efficacies from.
            side: Side length of the efficacy map.
            imside: Side length of the input image.
            cmap: Colormap for plotting group labels.
        """
        self.model = model
        self.loss_manager = loss_manager
        self.side = side
        self.imside = imside
        self.cmap = cmap
        plt.ion()
        plt.close("all")
        self.fig, self.ax = plt.subplots(1, 2, figsize=(6, 3))
        self.ax[0].set_axis_off()
        self.im = self.ax[0].imshow(np.zeros([imside, imside]), vmin=0, vmax=1)
        self.ax[0].set_title("Weights")
        self.ax[1].set_axis_off()
        self.efim = self.ax[1].imshow(np.zeros([side, side]), vmin=0, vmax=1)
        self.ax[0].set_title("Efficacy")
        plt.pause(0.1)

    def plot_weights_and_efficacies(self):
        """Plots weights and efficacies."""
        # Reshape the weights into a visualizable format
        w = (
            self.model.get_weights()
            .reshape(self.imside, self.imside, self.side, self.side)
            .transpose(2, 0, 3, 1)
            .reshape(self.imside * self.side, self.imside * self.side)
        )
        ef = self.loss_manager.get_efficacies().reshape(self.side, self.side)
        # Reshape group labels into the efficacy map shape
        self.im.set_array(w)
        self.efim.set_array(ef)
        plt.pause(0.1)


def initialize_mnist_loaders(tasks, params):
    print("Initialize MNIST dataset")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.flatten(x)),
            transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
        ]
    )
    mnist_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    mnist_test = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    data_labels = mnist_test.targets.unique()

    print("Initialize Task Dataset loaders")
    train_loaders, test_loaders = [], []
    for task in tasks:
        train_loader, test_loader = create_data_loaders(
            mnist_train, mnist_test, task, params.batch_size, params.subset
        )
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    return train_loaders, test_loaders, data_labels


def get_tasks(n_elems, task_width=2):
    c2 = torch.tensor(list(combinations(range(n_elems), task_width)))

    tasks = []
    while len(c2) > 0:
        n = len(c2)
        task = c2[torch.randperm(n)[0]]
        tasks.append(task)
        c2 = c2[[all([x not in task for x in c]) for c in c2]]

    return torch.stack(tasks).tolist()


def get_anchors(n_elems, params, device):
    n = 1000

    points = torch.rand((n, 2))
    kmeans = KMeans(n_clusters=n_elems, mode="euclidean")
    clusters = kmeans.fit_predict(points.unsqueeze(0))

    anchors = torch.stack(
        [points[clusters.flatten() == x].mean(0) for x in clusters.unique()]
    ).to(device) * np.sqrt(params.latent_dim)

    anchors = anchors[torch.randperm(len(anchors))]

    return anchors


def generate_color_vector(n_elems):
    colors = [
        "#FF5733",
        "#33FF57",
        "#5733FF",
        "#FFFF33",
        "#33FFFF",
        "#FF33FF",
        "#FF8000",
        "#8000FF",
        "#00FF80",
        "#808080",
        "#404040",
        "#000000",
    ]
    cmap = ListedColormap(colors)
    return cmap(np.linspace(0, 1, n_elems % len(colors)))


def setup_data(params, device):
    n_labels = 10
    tasks = get_tasks(n_labels)
    anchors = get_anchors(n_labels, params, device)
    cmap = generate_color_vector(n_labels)
    train_loaders, test_loaders, data_labels = initialize_mnist_loaders(
        tasks, params
    )
    return train_loaders, test_loaders, data_labels, anchors, cmap, tasks


def main(params, use_wandb=False, device="gpu"):
    """Main function to train and evaluate the model."""

    sys.stdout.flush()
    sys.stderr.flush()

    # Define the format for NumPy array printing.
    NP_FORMAT = {"float": "{:8.4f}".format}
    np.set_printoptions(formatter=NP_FORMAT, linewidth=999)

    train_loaders, test_loaders, data_labels, anchors, cmap, tasks = (
        setup_data(params, device)
    )

    # Initialize the model and optimizer.
    model, optimizer = initialize_models(
        params.input_dim, params.latent_dim, params.learning_rate
    )

    print("Initialize the LOSS manager")
    # Initialize the loss manager.
    lossManager = LossEfficacyFactory(
        model=model,
        mode="stm",
        efficacy_radial_sigma=params.efficacy_radial_sigma,
        efficacy_decay=params.efficacy_decay,
        efficacy_saturation_factor=params.efficacy_saturation_factor,
    ).to(device)

    print("Initialize plotting")
    # Initialize plotting utilities.
    side = model.radial.side
    imside = 28
    plotter = Plotter(model, lossManager, side, imside, cmap)

    regress_models = {}
    # Iterate through each task and train the model.
    for i, task in enumerate(tasks):
        print(f"Training on task {i+1}: {task}")
        train_stm(
            model,
            optimizer,
            train_loaders[i],
            lossManager,
            anchors,
            params.epochs,
            params.neigh_sigma_base,
            params.neigh_sigma_max,
            params.lr_base,
            params.lr_max,
            params.anchor_sigma,
            device,
        )

        # # REGRESS MODEL METHOD

        regress_model = get_regression(
            model,
            train_loaders[i : i + 1],
            device,
        )

        regress_models[i] = regress_model

        regress_accuracy = evaluate_model_by_regression(
            model,
            test_loaders[i: i + 1],
            regress_model,
            device,
        )
        plotter.plot_weights_and_efficacies()

        print(f"Regress accuracy on task {i+1}: {100 * regress_accuracy:.2f}%")

        if use_wandb:
            wandb.log(
                {
                    "task": {i + 1},
                    "accuracy": regress_accuracy,
                    "plot": plotter.fig,
                }
            )


# Define a function to parse command-line arguments.
def parse_arguments():
    """Parses command-line arguments for the simulation.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--opt",
        nargs=1,
        metavar="KEY=VALUE",
        action="append",
        help="Additional simulation option in KEY=VALUE format",
    )

    parser.add_argument(
        "-w",
        "--wandb",
        action="store_true",
        help="Use wandb",
    )

    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="sim",
        help="Use wandb",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="Set the seed for random number generation",
    )

    args = parser.parse_args()

    return args


def encode_parameter_string(opt):
    # Process additional options if provided.
    if opt is not None:
        opts = ";".join([s[0].strip() for s in opt])
        return opts


# %%
if __name__ == "__main__":

    # # %%
    #
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # seed = 0
    # use_wandb = False
    # name = "test"
    # opt = [["epochs=10"], ["anchor_sigma=1.2"], ["latent_dim=900"]]
    #
    # updated_params = encode_parameter_string(opt)
    # params = Parameters()
    # params.init_name = name
    # updated_json = params.update(updated_params)
    # generator = torch.manual_seed(seed)
    #
    # # %% TEST
    #
    # # Define the format for NumPy array printing.
    # NP_FORMAT = {"float": "{:8.4f}".format}
    # np.set_printoptions(formatter=NP_FORMAT, linewidth=999)
    #
    # train_loaders, test_loaders, data_labels, anchors, cmap, tasks = (
    #     setup_data(params, DEVICE)
    # )
    #
    # # Initialize the model and optimizer.
    # model, optimizer = initialize_models(
    #     params.input_dim, params.latent_dim, params.learning_rate
    # )
    #
    # print("Initialize the LOSS manager")
    # # Initialize the loss manager.
    # lossManager = LossEfficacyFactory(
    #     model=model,
    #     mode="stm",
    #     efficacy_radial_sigma=params.efficacy_radial_sigma,
    #     efficacy_decay=params.efficacy_decay,
    #     efficacy_saturation_factor=params.efficacy_saturation_factor,
    # ).to(DEVICE)
    #
    # i = 4
    # task = tasks[i]
    #
    # print(f"Training on task {i+1}: {task}")
    # train_stm(
    #     model,
    #     optimizer,
    #     train_loaders[i],
    #     lossManager,
    #     anchors,
    #     params.epochs,
    #     params.neigh_sigma_base,
    #     params.neigh_sigma_max,
    #     params.lr_base,
    #     params.lr_max,
    #     params.anchor_sigma,
    #     DEVICE,
    # )
    # # %%
    # targets = []
    # norms = []
    # for data, target in train_loaders[i]:
    #     data, target = data.to(DEVICE), target.to(DEVICE)
    #     norm = model(data)
    #     targets.append(target.cpu().detach().numpy())
    #     norms.append(norm.cpu().detach().numpy())
    # # %%
    # targets = np.hstack(targets)
    # norms = np.vstack(norms)
    #
    # ltargets = np.unique(targets)
    #
    # n_zeros = norms[targets == ltargets[0]]
    # n_ones = norms[targets == ltargets[1]]
    # s_zeros = softmax(1e3 / n_zeros, 1)
    # s_ones = softmax(1e3 / n_ones, 1)
    # w = np.array([s_zeros.mean(0), s_ones.mean(0)])
    #
    # # %%
    # targets = []
    # norms = []
    # datas = []
    # for data, target in test_loaders[i]:
    #     data, target = data.to(DEVICE), target.to(DEVICE)
    #     norm = model(data)
    #     targets.append(target.cpu().detach().numpy())
    #     norms.append(norm.cpu().detach().numpy())
    #     datas.append(data.cpu().detach().numpy())
    #
    # targets = np.hstack(targets)
    # norms = np.vstack(norms)
    # datas = np.vstack(datas)
    #
    # ltargets = np.unique(targets)
    #
    # n_zeros = norms[targets == ltargets[0]]
    # n_ones = norms[targets == ltargets[1]]
    # s_zeros = softmax(1e3 / n_zeros, 1)
    # s_ones = softmax(1e3 / n_ones, 1)
    #
    # softmax(1e5 * (s_ones @ w.T), axis=1).sum(axis=0)
    # tot = datas.shape[0]
    # accurate = softmax(1e5 * (s_zeros @ w.T), axis=1).sum(axis=0)[0]
    # accurate += softmax(1e5 * (s_ones @ w.T), axis=1).sum(axis=0)[1]
    #
    # print(f"accuracy on task {i} is {100*accurate/tot:<5.2f}%")
    #
    # # %%
    #
    # # %%

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_arguments()
    seed = args.seed
    use_wandb = args.wandb
    name = args.name
    opt = args.opt

    updated_params = encode_parameter_string(opt)
    params = Parameters()
    params.init_name = name
    updated_json = params.update(updated_params)
    torch.manual_seed(seed)

    if use_wandb:
        matplotlib.use("agg")
        wandb.init(
            project=params.project_name,
            entity=params.entity_name,
            name=params.init_name,
        )

        wandb.log(updated_json)

    main(params, use_wandb=use_wandb, device=DEVICE)

    if use_wandb:
        wandb.finish()
