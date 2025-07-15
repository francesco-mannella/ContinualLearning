import json
import sys


class ParameterManager:

    def __init__(self):
        self._set_param_types()

    def _set_param_types(self):
        self.param_types = {
            k: type(v)
            for k, v in self.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }

    def check_json_format(self, json_string):
        json_string = (
            json_string.replace("'", '"')
            .replace(" ", "")
            .replace("True", "true")
            .replace("False", "false")
        )
        return json_string

    def _string_to_json(self, param_string, mode="user"):
        """Converts a semicolon-separated string to a JSON dictionary.

        Args:
            param_string (str): String representing the params.
            mode (str): Format of the string; "user" for
              semicolon-separated key-value pairs, "json" for JSON string.

        Returns:
            dict: Dictionary representing the JSON object.

        """
        if not param_string:
            return {}

        if mode == "user":
            # Split the string into key-value pairs
            param_string = param_string.replace(" ", "")
            params = dict(
                s.split("=", 1) for s in param_string.split(";") if "=" in s
            )
            # Format the key-value pairs into a JSON string
            params = ",".join(f'"{k.strip()}":{v}' for k, v in params.items())
            params = "{" + params + "}"
        else:
            params = param_string

        params = self.check_json_format(params)

        try:
            param_dict = json.loads(params)
        except ValueError as e:
            print(f"Error decoding JSON: {e}")
            sys.exit(1)

        return param_dict

    def _json_to_params(self, param_dict):
        """Set attributes from a dictionary.

        Args:
            param_dict (dict): Dictionary of parameters.
        """
        # Iterate over the key-value pairs in the input dictionary.
        for key, value in param_dict.items():
            # For each pair, set an attribute of the instance.
            setattr(self, key, value)

    def _params_to_dict(self):
        params = {
            key: value
            for key, value in self.__dict__.items()
            if key != "param_types" and not callable(value)
        }
        return params

    def __repr__(self):
        params = self._params_to_dict()
        return json.dumps(params)

    def update(self, param_string, mode="user"):
        """Update parameters based on the input string.

        Args:
            param_string (str): Input string containing parameters.
            mode (str, optional): "user" to convert the input
                string to JSON. "json" if the input string is a
                json format. Defaults to "user".
        """
        if mode == "user":
            # Convert user-provided string to JSON format
            param_dict = self._string_to_json(param_string)
        elif mode == "json":
            param_dict = self._string_to_json(param_string, mode="json")

        # Update internal parameters using the JSON dictionary
        self._json_to_params(param_dict)

    def save(self, filepath, mode="user"):
        """Saves parameters to a file.

        Args:
            filepath (str): The path to the file.
            mode (str, optional): Specifies the saving mode.
              "user": Saves parameters in a human-readable format.
              Each parameter is written as 'key = value' on a new line.
              "json": Saves parameters in JSON format.
              Defaults to "user".
        """
        with open(filepath, "w") as file:
            if mode == "user":
                for key, value in self._params_to_dict().items():
                    file.write(f"{key} = {value}\n")
            elif mode == "json":
                params = self._params_to_dict()
                json.dump(params, file, indent=4)

    def load(self, filepath, mode="user"):
        """Loads parameters from a file.

        Args:
            filepath (str): The path to the file.
            mode (str, optional): Specifies the loading mode.
              "user": Loads parameters from a human-readable format.
              Expects each parameter to be in the format 'key = value'.
              "json": Loads parameters from JSON format.
              Defaults to "user".
        """
        with open(filepath, "r") as file:
            if mode == "user":
                param_list = "".join([line.strip() + ";" for line in file])
                self.update(param_list)
            elif mode == "json":
                params = json.load(file)
                self.__dict__.update(params)

    def __hash__(self):
        # Using a tuple comprehension to collect all non-callable and
        # non-private attributes (those not starting with "_") into a tuple
        attr_values = tuple(
            (attr, self._make_hashable(getattr(self, attr)))
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("_")
        )
        hashid = hash(attr_values)
        # Create a unique string from the tuple and return its hash
        return hashid

    def _make_hashable(self, value):
        if isinstance(value, dict):
            # Convert dictionary to a frozenset of its items (key-value pairs)
            return frozenset(
                (key, self._make_hashable(v)) for key, v in value.items()
            )
        elif isinstance(value, list):
            # Convert list to a tuple of its elements
            return tuple(self._make_hashable(v) for v in value)
        elif isinstance(value, set):
            # Convert set to a frozenset of its elements
            return frozenset(self._make_hashable(v) for v in value)
        # Add other types like list, set, etc., if needed
        return value


class Parameters(ParameterManager):
    def __init__(
        self,
        learning_rate=0.001,
        batch_size=64,
        epochs=100,
        subset=-1,
        input_dim=784,
        latent_dim=400,
        anchor_sigma=1.4,
        neigh_sigma_max=40,
        neigh_sigma_base=0.7,
        lr_max=2,
        lr_base=0.001,
        efficacy_radial_sigma=10,
        efficacy_decay=0.005,
        efficacy_saturation_factor=2.5,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.subset = subset
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.anchor_sigma = anchor_sigma
        self.neigh_sigma_max = neigh_sigma_max
        self.neigh_sigma_base = neigh_sigma_base
        self.lr_max = lr_max
        self.lr_base = lr_base
        self.efficacy_radial_sigma = efficacy_radial_sigma
        self.efficacy_decay = efficacy_decay
        self.efficacy_saturation_factor = efficacy_saturation_factor

        super(Parameters, self).__init__()


if __name__ == "__main__":

    params = Parameters()

    params.update("batch_size=128")

    params.save("/tmp/params")

    params1 = Parameters()

    params1.load("/tmp/params")
