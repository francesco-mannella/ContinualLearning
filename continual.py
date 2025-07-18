import argparse
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import tqdm
from kmeans_pytorch import kmeans as KMeans
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import wandb
from params import Parameters
from topological_maps import LossEfficacyFactory, TopologicalMap


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


def get_regress_matrix(model, loaders):
    """Calculates the regression matrix based on model outputs.

    Args:
        model: The trained model.
        loaders: List of data loaders to process.

    Returns:
        numpy.ndarray: The regression matrix.
    """
    resps = []
    for loader in loaders:
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            norms = model(data)
            latent = torch.softmax(1000 / norms, 1).round()
            resps.append(
                np.hstack(
                    [
                        to_numpy(target).reshape(-1, 1),
                        to_numpy(latent).reshape(-1, model.output_size),
                    ]
                )
            )

    resps = pd.DataFrame(
        np.vstack(resps),
        columns=["target", *np.arange(model.side**2)],
    )

    # Group by target and calculate mean responses
    regres_w = resps.groupby("target").mean().reset_index().to_numpy()
    regres_w[:, 1:] /= regres_w[:, 1:].sum(1).reshape(-1, 1)

    return regres_w


def get_regress_matrix_new(model, loaders, regres_w):
    """Calculates the regression matrix based on model outputs.

    Args:
        model: The trained model.
        loaders: List of data loaders to process.
        regres_w: regression matrix for all labels

    Returns:
        numpy.ndarray: The regression matrix.
    """
    resps = []
    for loader in loaders:
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            norms = model(data)
            latent = torch.softmax(1000 / norms, 1).round()
            resps.append(
                np.hstack(
                    [
                        to_numpy(target).reshape(-1, 1),
                        to_numpy(latent).reshape(-1, model.output_size),
                    ]
                )
            )

    resps = pd.DataFrame(
        np.vstack(resps),
        columns=["target", *np.arange(model.side**2)],
    )

    # Group by target and calculate mean responses
    curr_regres_w = resps.groupby("target").mean().reset_index().to_numpy()
    curr_regres_w[:, 1:] /= regres_w[:, 1:].sum(1).reshape(-1, 1)
    regres_w[curr_regres_w[:, 0].flatten().int()] = curr_regres_w[:, 1:]

    return regres_w


def evaluate_model(model, test_loaders, w_regress, DEVICE):
    """Evaluates the model accuracy.

    Args:
        model: The model to evaluate.
        test_loaders: List of test data loaders.
        w_regress: Regression weights.
        DEVICE: Device to use (CPU or GPU).

    Returns:
        The accuracy of the model.
    """
    correct = 0
    total = 0
    regress = w_regress[:, 1:]
    with torch.no_grad():
        for test_loader in test_loaders:
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                norms = model(data)
                latent = torch.softmax(1000 / norms, 1)
                predicted = (to_numpy(latent) @ regress.T).argmax(1)
                total += target.size(0)
                correct += (predicted == to_numpy(target)).sum()

    accuracy = correct / total

    return accuracy


def get_groups(model, lossManager, n_groups):
    """Clusters weights into groups using K-means.

    Args:
        model: The model to get weights from.
        lossManager: an object ith info about loss and efficacy
        n_groups: The number of groups to cluster into.

    Returns:
        Group labels and centroids.
    """
    weights = model.get_weights("torch").T
    weights = torch.nn.functional.normalize(weights, 0)

    efficacy = lossManager.get_efficacies()
    efficacy = torch.tensor(efficacy).to(weights.device)
    efficacy_idcs = torch.where(efficacy > 0.3)[0]

    group_labels = (n_groups + 1) * torch.ones(weights.shape[0]).long()
    group_labels[efficacy_idcs], group_centroids = KMeans(
        weights[efficacy_idcs],
        num_clusters=n_groups,
        tol=1e-30,
        device=DEVICE,
    )

    return group_labels, group_centroids


def get_ordered_labels(labels, anchors):
    side = int(np.sqrt(len(labels)))
    t = torch.arange(side, device=labels.device)
    x, y = torch.meshgrid(t, t, indexing="ij")
    indices = torch.stack([x.flatten(), y.flatten()], dim=1).to(labels.device)
    ianchors = anchors.round().int().to(labels.device)
    labels_names = labels.unique()

    dists = torch.cdist(indices.float(), ianchors.float())
    anchor_in_labels_idcs = dists.argmin(0)
    labels_order = labels[anchor_in_labels_idcs]
    not_anch_labels = torch.where(
        torch.tensor([x not in labels_order for x in labels_names]).to(
            labels.device
        )
    )[0]
    labels_order = torch.hstack(
        [labels_order, labels_names[not_anch_labels].flatten()]
    )

    return labels_order[labels]


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
        self.fig, self.ax = plt.subplots(1, 4, figsize=(9, 2))
        self.ax[0].set_axis_off()
        self.im = self.ax[0].imshow(np.zeros([imside, imside]), vmin=0, vmax=1)
        self.ax[1].set_axis_off()
        self.efim = self.ax[1].imshow(np.zeros([side, side]), vmin=0, vmax=1)
        self.ax[2].set_axis_off()
        self.clim = self.ax[2].imshow(np.zeros([side, side, 3]))
        self.ax[3].set_axis_off()
        # Reshape and tile the colormap for display as a colorbar
        self.bar = self.ax[3].imshow(
            cmap.reshape(-1, 1, 4) * (np.ones([2, 4]).reshape(1, 2, 4))
        )
        plt.pause(0.1)

    def plot_weights_and_efficacies(self, group_labels):
        """Plots weights, efficacies, and group labels.

        Args:
            group_labels: Cluster labels for each neuron.
        """
        # Reshape the weights into a visualizable format
        w = (
            self.model.get_weights()
            .reshape(self.imside, self.imside, self.side, self.side)
            .transpose(2, 0, 3, 1)
            .reshape(self.imside * self.side, self.imside * self.side)
        )
        ef = self.loss_manager.get_efficacies().reshape(self.side, self.side)
        # Reshape group labels into the efficacy map shape
        cl = group_labels.reshape(20, 20).detach().cpu().numpy()
        cl = self.cmap[cl]
        self.im.set_array(w)
        self.efim.set_array(ef)
        self.clim.set_array(cl)
        plt.pause(0.1)


def main(params, use_wandb=False):
    """Main function to train and evaluate the model."""

    sys.stdout.flush()
    sys.stderr.flush()

    # Define the format for NumPy array printing.
    NP_FORMAT = {"float": "{:8.4f}".format}
    np.set_printoptions(formatter=NP_FORMAT, linewidth=999)

    # Define the tasks as pairs of digits.
    tasks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    # Define anchor points and scale them based on latent dimension.
    anchors = torch.tensor(
        [
            [0.15, 0.17],
            [0.12, 0.54],
            [0.16, 0.84],
            [0.50, 0.15],
            [0.36, 0.45],
            [0.62, 0.50],
            [0.48, 0.82],
            [0.83, 0.17],
            [0.88, 0.50],
            [0.83, 0.83],
            [9.00, 9.00],
        ]
    ).to(DEVICE) * np.sqrt(params.latent_dim)

    # Define colormap for group labels.
    cmap = ListedColormap(
        [
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
    )(np.linspace(0, 1, 11))

    print("Initialize MNIST dataset")
    # Define transformations to apply to the MNIST dataset.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.flatten(x)),
            transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
        ]
    )
    # Load the MNIST training and testing datasets.
    mnist_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    mnist_test = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    print("Initialize Task Dataset loaders")
    train_loaders, test_loaders = [], []
    # Create data loaders for each task.
    for task in tasks:
        train_loader, test_loader = create_data_loaders(
            mnist_train, mnist_test, task, params.batch_size, params.subset
        )
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

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
    ).to(DEVICE)

    print("Initialize plotting")
    # Initialize plotting utilities.
    side = model.radial.side
    imside = 28
    plotter = Plotter(model, lossManager, side, imside, cmap)

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
            DEVICE,
        )

        # Evaluate the model and plot the results.
        w_regress = get_regress_matrix(model, train_loaders[: i + 1])
        accuracy = evaluate_model(
            model,
            test_loaders[: i + 1],
            w_regress,
            DEVICE,
        )

        # Determine number of groups for current task
        group_labels, _ = get_groups(model, lossManager, (i + 1) * 2)
        ordered_group_labels = get_ordered_labels(group_labels, anchors)
        plotter.plot_weights_and_efficacies(ordered_group_labels)

        print(f"Accuracy on task {i+1}: {100 * accuracy:.2f}%")
        wandb.log(
            {
                "task": {i + 1},
                "accuracy": accuracy,
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
        opts = ";".join([s[0] for s in args.opt])
        return opts


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_arguments()
    updated_params = encode_parameter_string(args.opt)
    seed = args.seed
    use_wandb = args.wandb
    name = args.name

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

    main(params, use_wandb=use_wandb)

    if use_wandb:
        wandb.finish()
