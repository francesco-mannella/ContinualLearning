import math

import kmeans_pytorch
import torch


class DimensionalityError(Exception):
    def __str__(self):
        return "Dimensionality of the output must be 1D or 2D."


class RadialBasis(torch.nn.Module):
    """
    This code creates a radial grid in either 1D or 2D based on a given
    centroid. It can be used to generate a grid of points around a central
    point.
    """

    def __init__(self, size, dims):
        """
        This function creates a radial grid with a given size and
        dimensionality.

        Args:
            size (int): The size of the radial grid.
            dims (int): The dimensionality of the grid (1 for 1D, 2 for 2D).

        """
        super(RadialBasis, self).__init__()
        self.dims = dims
        self.size = size

        if self.dims == 1:
            self.grid = torch.arange(self.size)
            self.side = self.size
        elif self.dims == 2:
            self.side = int(math.sqrt(self.size))
            if self.side**2 != self.size:
                raise "Dimensions must be equal"
            t = torch.arange(self.side)
            meshgrids = torch.meshgrid(t, t, indexing="ij")
            self.grid = torch.stack([x.reshape(-1) for x in meshgrids]).T
        else:
            raise DimensionalityError()
        self.grid = self.grid.unsqueeze(dim=0).float()

    def _apply(self, fn):
        """Applies a function recursively to all tensors.

        Applies to sub-modules, direct attributes, and the `radial`
        attribute (assumed to be a `nn.Module`).  This ensures
        changes like device or data type are propagated correctly.

        Args:
            fn (function): The function to apply to the tensors.

        Returns:
            nn.Module: The modified module.
        """
        super(RadialBasis, self)._apply(fn)
        self.grid = fn(self.grid)
        return self

    def forward(self, index, std, as_point=False):
        """
        Args:
            - index (int): Indicates the point at the center of the Gaussian on
              the flattened grid, which is arranged in rows.
            - std (float): The standard deviation of the function.
            - as_point (bool, optional): Whether to treat index as a point or
              not.  Defaults to False.

        Returns:
            The result of the function call.
        """

        if self.dims == 1:
            x = index.unsqueeze(dim=-1)
            dists = self.grid - x
        elif self.dims == 2:
            if as_point:
                x = index.unsqueeze(dim=1)
            else:
                row = index // self.side
                col = index % self.side
                x = torch.stack([row, col]).T.to(device=self.grid.device)
                x = x.unsqueeze(dim=1)
            dists = torch.norm(self.grid - x, dim=-1)

        if torch.is_tensor(std):
            std = std.to(self.grid.device)

        output = torch.exp(-0.5 * (std**-2) * dists**2)
        output /= output.sum(dim=-1).unsqueeze(dim=-1)

        return output


class TopologicalMap(torch.nn.Module):
    """
    A neural network module that represents a topological map. This map models
    data topology through connected nodes
    """

    def __init__(
        self, input_size, output_size, output_dims=2, parameters=None
    ):
        """Initializes the TopologicalMap.

        Args:
            input_size (int): Number of inputs.
            output_size (int): Number of outputs.
            output_dims (int, optional): Output space dimensions.
                Defaults to 2.
            parameters (numpy.ndarray, optional): Initial weights.
                Defaults to None (Xavier initialization).
        """

        super(TopologicalMap, self).__init__()

        if parameters is None:
            weights = torch.empty(input_size, output_size)
            torch.nn.init.xavier_normal_(weights)
            self.weights = torch.nn.Parameter(
                1e-4 * weights, requires_grad=True
            )
        else:
            parameters = torch.tensor(parameters).float()
            self.weights = torch.nn.Parameter(parameters, requires_grad=True)

        self.input_size = input_size
        self.output_size = output_size
        self.output_dims = output_dims
        self.radial = RadialBasis(output_size, output_dims)
        self.std_init = (
            self.output_size
            if output_dims == 1
            else int(math.sqrt(self.output_size))
        )
        self.curr_std = self.std_init
        self.bmu = None
        self.side = (
            None if output_dims == 1 else int(math.sqrt(self.output_size))
        )

    def _apply(self, fn):
        """Applies a function recursively to all tensors.

        Applies to sub-modules or direct attributes of this module.
        Overrides the `_apply` method of `nn.Module`.
        Applies `fn` to the `radial` attribute (a `nn.Module`).

        Args:
            fn: The function to apply to the module's tensors.

        Returns:
            TopologicalMap: self.
        """
        super(TopologicalMap, self)._apply(fn)
        self.radial = self.radial._apply(fn)
        return self

    def forward(self, x):
        """
        Computes the forward pass by calculating squared Euclidean distances
        between input data and the network's weights.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Squared Euclidean distances for each weight vector.
        """

        diffs = self.weights.unsqueeze(dim=0) - x.unsqueeze(dim=-1)

        # Compute the Euclidean norms of the differences along dimension 1
        norms = torch.norm(diffs, dim=1)

        # Square the norms to obtain squared distances
        norms2 = torch.pow(norms, 2)

        return norms2

    def find_bmu(self, x):
        """
        Identifies the index of the Best Matching Unit (BMU) for a given input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Index of the BMU.
        """
        return torch.argmin(x, dim=-1).detach()

    def get_representation(self, x, rtype="point", neighborhood_std=None):
        """Generates the representation of the Best Matching Unit (BMU).

        Args:
            x (torch.Tensor): Input tensor.
            rtype (str, optional): Representation type ('point', 'grid').
                Defaults to 'point'.
            neighborhood_std (float, optional): Neighborhood std dev.
                Defaults to current neighborhood_std.

        Returns:
            torch.Tensor or None: BMU representation; None if unavailable.
        """

        self.bmu = self.find_bmu(x)
        if self.bmu is not None:
            if rtype == "point":
                if self.output_dims == 1:
                    return self.bmu.float()

                elif self.output_dims == 2:
                    row = self.bmu // self.side
                    col = self.bmu % self.side
                    return torch.stack([row, col]).T.float()

            elif rtype == "grid":
                if neighborhood_std is None:
                    neighborhood_std = self.curr_neighborhood_std
                phi = self.radial(self.bmu, neighborhood_std)
                return phi
        else:
            return None

    def backward(self, point, neighborhood_std=None):
        """Executes the backward pass for a specified point.

        Args:
            point (int, int): Target point for the backward pass.
            neighborhood_std (float, optional): Standard deviation for
                the radial basis function.
                Defaults to current neighborhood_std.

        Returns:
            torch.Tensor: Result of the backward pass for the point.
        """

        if neighborhood_std is None:
            neighborhood_std = self.curr_neighborhood_std
        phi = self.radial(point, neighborhood_std, as_point=True)
        max_idx = phi.argmax()
        hov_phi = torch.nn.functional.one_hot(
            max_idx, self.output_size
        ).float()
        output = torch.matmul(hov_phi, self.weights.T)
        return output

    def get_weights(self, mode="np"):
        """Returns the weights of the model.

        Args:
            mode (str, optional): Specifies the format of the returned
              weights. It can be 'np' for numpy array or 'torch' for
              torch tensor. Defaults to "np".

        Returns:
            np.ndarray: The weights of the model as a numpy array.
        """
        # Extract weights from the model and detach from the computation graph
        w = self.weights.detach()
        if mode == "np":
            return w.cpu().numpy()
        elif mode == "torch":
            return w

    def get_anchor_groups(self, anchors):
        """Assigns each weight vector to an anchor group using k-means.

        Args:
            anchors (torch.Tensor): Anchor points tensor.

        Returns:
            torch.Tensor: Anchor group assignments for each weight vector.
        """

        cluster_ids, cluster_centers = kmeans_pytorch.kmeans(
            X=self.model.weights.T,
            num_clusters=len(anchors),
            distance="euclidean",
            device=self.model.device,
        )

        side_length = self.model.radial.side
        side_indices = torch.arange(side_length)
        # Stack the indices and cluster IDs
        coordinate_cluster_ids = torch.stack(
            [
                *[
                    X.flatten()
                    for X in torch.meshgrid(side_indices, side_indices)
                ],
                cluster_ids,
            ]
        ).T

        # Calculate the mean coordinate for each cluster
        cluster_means = torch.stack(
            [
                coordinate_cluster_ids[coordinate_cluster_ids[:, 2] == x]
                .float()
                .mean(0)
                for x in range(len(anchors))
            ]
        )

        # Assign each cluster to the nearest anchor
        cluster_to_anchor = (
            torch.norm(
                anchors.cpu().reshape(-1, 1, 2)
                - cluster_means[:, :2].reshape(1, -1, 2),
                dim=-1,
            )
            .min(0)
            .indices
        )

        # Assign each weight vector to the anchor group of its cluster
        anchor_groups = cluster_to_anchor[coordinate_cluster_ids[:, 2]]

        return anchor_groups


class LossFactory:
    """Builds the loss function for a SOM or STM model.

    Args:
        model (torch.nn.Module): The SOM or STM model to update.
        mode (str): Specifies the update type ('som' or 'stm').
        kernel_function (callable, optional): Kernel function.
          Defaults to:
            - ``lambda phi: phi`` if mode is 'som'.
            - ``lambda phi, psi: phi * psi`` if mode is 'stm'.
    """

    def __init__(
        self,
        model,
        mode="som",
        kernel_function=None,
    ):
        """Initializes LossFactory with model, mode, and kernel function."""
        self.model = model
        self.mode = mode

        # Determine the kernel function based on the mode if not provided.
        if kernel_function is None:
            if self.mode == "som":
                self.kernel_function = lambda phi: phi
            elif self.mode == "stm":
                self.kernel_function = lambda phi, psi: phi * psi
        else:
            self.kernel_function = kernel_function

    def get_weighted_norms(
        self,
        bmu,
        norms2,
        neighborhood_std,
        anchors=None,
        neighborhood_std_anchors=None,
    ):
        """Compute the SOM/STM loss.

        Args:
            bmu (array-like): Indices of best matching units.
            norms2 (array-like): Squared norms between weights/outputs.
            neighborhood_std (float): Std deviation for neighborhood.
            anchors (array-like, optional): Labels/anchors for modulation.
                Defaults to None.
            neighborhood_std_anchors (float, optional): Std deviation for
                anchors neighborhood modulation. Defaults to neighborhood_std.

        Returns:
            array-like: The values of the computed losses.
        """
        self.model.curr_neighborhood_std = neighborhood_std

        if anchors is None:
            # Calculate neighborhood influence using radial basis function.
            phi = self.model.radial(bmu, neighborhood_std)
            # Compute loss without anchor modulation.
            _losses = 0.5 * norms2 * self.kernel_function(phi)
        else:
            # Use anchors for neighborhood modulation if provided.
            if neighborhood_std_anchors is None:
                neighborhood_std_anchors = neighborhood_std

            phi = self.model.radial(bmu, neighborhood_std)
            psi = self.model.radial(
                anchors, neighborhood_std_anchors, as_point=True
            )
            self.kernel = self.kernel_function(phi, psi)
            _losses = 0.5 * norms2 * self.kernel

        return _losses

    def losses(
        self,
        norms2,
        neighborhood_std,
        anchors=None,
        neighborhood_std_anchors=None,
    ):
        """Compute the SOM/STM loss.

        Args:
            norms2 (array-like): Squared norms between weights and
              outputs.
            neighborhood_std (float): Standard deviation for neighborhood
              radial calculation.
            anchors (array-like, optional): Labels/anchors for
              neighborhood modulation. Default is None.
            neighborhood_std_anchors (float, optional): Standard
              deviation for anchors neighborhood modulation. Defaults to
              neighborhood_std.

        Returns:
            array-like: Computed loss values.
        """
        # Find best matching unit indices based on norms2.
        self.model.bmu = self.model.find_bmu(norms2)

        # Calculate weighted norms using BMU and other parameters.
        _losses = self.get_weighted_norms(
            self.model.bmu,
            norms2,
            neighborhood_std,
            anchors,
            neighborhood_std_anchors,
        )

        return _losses


class LossEfficacyFactory(LossFactory):
    """
    A class to manage loss computation with efficacy modulation.

    This class extends LossFactory and incorporates an efficacy mechanism
    that adjusts the loss based on how effectively each prototype
    represents the input data. It uses radial basis functions (RBFs)
    and a decay mechanism to update the efficacies of prototypes over time.

    Attributes:
        efficacy_radial_sigma (float): Sigma for the radial basis function
            used to compute efficacy.
        efficacy_decay (float): Decay rate for updating prototype efficacies.
        _efficacies (torch.Tensor): Efficacy values for each prototype.
        _inefficacies (torch.Tensor): Inefficacy values for each prototype.
    """

    def __init__(
        self,
        efficacy_radial_sigma,
        efficacy_decay,
        efficacy_saturation_factor,
        *args,
        **kwargs,
    ):
        """
        Initializes LossEfficacyFactory with efficacy parameters.

        Args:
            efficacy_radial_sigma (float): Sigma for the radial basis
                function.
            efficacy_decay (float): Decay rate for updating prototype
                efficacies.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(LossEfficacyFactory, self).__init__(*args, **kwargs)
        self.efficacy_radial_sigma = efficacy_radial_sigma
        self.efficacy_decay = efficacy_decay
        self.efficacy_saturation_factor = efficacy_saturation_factor
        self._efficacies = torch.zeros(self.model.output_size)
        self._inefficacies = 1.0 - torch.zeros(self.model.output_size)

    def to(self, device):
        """
        Moves efficacy tensors to the specified device.

        Args:
            device (torch.device): The device to move the tensors to.

        Returns:
            self: The LossEfficacyFactory instance.
        """
        self._efficacies = self._efficacies.to(device)
        self._inefficacies = self._inefficacies.to(device)
        return self

    def get_efficacies(self):
        return self._efficacies.cpu().detach().numpy()

    def loss(
        self,
        norms2,
        neighborhood_baseline,
        neighborhood_max,
        modulation_baseline,
        modulation_max,
        anchors=None,
        neighborhood_std_anchors=None,
    ):
        """
        Computes the efficacy-modulated loss.

        This method calculates the loss by modulating neighborhood and
        modulation rates based on prototype efficacies.

        Args:
            norms2 (torch.Tensor): Squared norms of input vectors.
            neighborhood_baseline (float): Baseline neighborhood std value.
            neighborhood_max (float): Maximum neighborhood std value.
            modulation_baseline (float): Baseline modulation rate.
            modulation_max (float): Maximum modulation rate.
            anchors (torch.Tensor, optional): Anchors for loss
                calculation. Defaults to None.
            neighborhood_std_anchors (torch.Tensor, optional):
                Neighborhood std for anchors. Defaults to None.

        Returns:
            torch.Tensor: The mean efficacy-modulated loss.
        """
        # Find the Best Matching Unit (BMU) for each input vector
        self.model.bmu = self.model.find_bmu(norms2)

        with torch.no_grad():
            # Create a mask identifying the BMU for each vector in the batch
            batch_size = len(norms2)
            mask = torch.zeros_like(norms2)
            mask[torch.arange(batch_size), self.model.bmu] = 1

            # Compute radial basis functions (RBFs) from squared norms,
            # centered at zero. Only BMUs' RBFs are considered.
            norm_radial_bases = (
                torch.exp(
                    -0.5 * (self.efficacy_radial_sigma**-2) * norms2
                )  # Apply RBF formula.
                * mask
            )  # Mask non-BMU prototypes

            # Compute the mean RBF activation for each unit based on BMUs.
            mask_props = (mask > 0).sum(0).float()  # Count BMUs for each unit
            mask_props[mask_props == 0] = 1e-5  # Avoid division by zero
            norm_radial_bases = (
                norm_radial_bases * (mask / mask_props.reshape(1, -1))
            ).sum(
                0
            )  # Normalize and average

            # Update prototype efficacies as leakies of mean RBFs of BMUs.
            # Update only for units where there are BMUs in that batch
            mask_radials = norm_radial_bases != 0
            self._efficacies = (
                self._efficacies
                + self.efficacy_decay
                * (norm_radial_bases - self._efficacies)
                * mask_radials
            )

            # use tanh to saturate inefficacies
            self._inefficacies = 1.0 - torch.tanh(
                self.efficacy_saturation_factor * self._efficacies
            )

            # Reshape inefficacies to match batch size.
            # Each item has one inefficiency value.
            episode_inefficacies = (
                mask @ self._inefficacies.flatten()
            ).reshape(-1, 1)

            _neighborhood_std = (
                neighborhood_baseline
                + episode_inefficacies
                * (neighborhood_max - neighborhood_baseline)
            )

            _modulation_rate = modulation_baseline + episode_inefficacies * (
                modulation_max - modulation_baseline
            )

        losses = self.get_weighted_norms(
            self.model.bmu,
            norms2,
            _neighborhood_std,
            anchors,
            neighborhood_std_anchors,
        )

        _loss = losses * _modulation_rate

        return _loss.mean()


class Updater(LossFactory):
    """Update SOM or STM model.

    Args:
        model (torch.nn.Module): SOM or STM model to update.
        learning_rate (float): Optimizer learning rate.
        mode (str): Update type, 'som' or 'stm'.
        kernel_function (callable, optional): Kernel function.
          Defaults to:
            - ``lambda phi: phi`` if mode is 'som'.
            - ``lambda phi, psi: phi * psi`` if mode is 'stm'.
    """

    def __init__(self, model, learning_rate, mode="som", kernel_function=None):

        super().__init__(model, mode, kernel_function)

        self.optimizer = torch.optim.Adam(
            params=model.parameters(), lr=learning_rate
        )

    def __call__(
        self,
        output,
        neighborhood_std,
        learning_modulation,
        anchors=None,
        neighborhood_std_anchors=None,
    ):
        if self.mode == "som":
            losses = self.loss(output, neighborhood_std)
        elif self.mode == "stm":
            losses = self.losses(
                output, neighborhood_std, anchors, neighborhood_std_anchors
            )
        else:
            raise ValueError("Invalid mode. Use 'som' or 'stm'.")

        loss = (learning_modulation * losses).mean()
        modulated_loss = losses.mean()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss, modulated_loss
