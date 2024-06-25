import numpy as np
from __future__ import print_function

class RBM:
    def __init__(self, num_visible, num_hidden):
        """
        Initialize the RBM with the specified number of visible and hidden units.

        Parameters
        ----------
        num_visible: int
            Number of visible units.
        num_hidden: int
            Number of hidden units.
        """
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.debug_print = True

        # Initialize weights with a uniform distribution between -sqrt(6. / (num_hidden + num_visible))
        # and sqrt(6. / (num_hidden + num_visible)). This range is derived from the work by 
        # Xavier Glorot and Yoshua Bengio on the difficulty of training deep feedforward neural networks.
        np_rng = np.random.RandomState(1234)
        self.weights = np.asarray(np_rng.uniform(
            low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            size=(num_visible, num_hidden)
        ))

        # Add bias weights by inserting an extra row and column at the beginning.
        self.weights = np.insert(self.weights, 0, 0, axis=0)
        self.weights = np.insert(self.weights, 0, 0, axis=1)

    def train(self, data, max_epochs=1000, learning_rate=0.1):
        """
        Train the RBM using Contrastive Divergence.

        Parameters
        ----------
        data: np.ndarray
            A matrix where each row is a training example consisting of the states of visible units.
        max_epochs: int, optional
            Maximum number of epochs to train. Default is 1000.
        learning_rate: float, optional
            Learning rate for weight updates. Default is 0.1.

        Returns
        -------
        float
            The final reconstruction error.
        """
        num_examples = data.shape[0]

        # Insert bias units of 1 into the first column of the data matrix.
        data = np.insert(data, 0, 1, axis=1)

        for epoch in range(max_epochs):
            # Positive phase: Clamp the data and sample from the hidden units.
            pos_hidden_activations = np.dot(data, self.weights)
            pos_hidden_probs = self._logistic(pos_hidden_activations)
            pos_hidden_probs[:, 0] = 1  # Fix the bias unit.
            pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
            pos_associations = np.dot(data.T, pos_hidden_probs)

            # Negative phase: Reconstruct visible units and sample again from hidden units.
            neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
            neg_visible_probs = self._logistic(neg_visible_activations)
            neg_visible_probs[:, 0] = 1  # Fix the bias unit.
            neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
            neg_hidden_probs = self._logistic(neg_hidden_activations)
            neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

            # Update weights.
            self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)

            # Compute reconstruction error.
            error = np.sum((data - neg_visible_probs) ** 2)
            if self.debug_print:
                print(f"Epoch {epoch}: error is {error}")

        return error

    def run_visible(self, data):
        """
        Run the network on a set of visible units to get a sample of the hidden units.

        Parameters
        ----------
        data: np.ndarray
            A matrix where each row consists of the states of the visible units.

        Returns
        -------
        np.ndarray
            A matrix where each row consists of the hidden units activated from the visible units in the data matrix.
        """
        num_examples = data.shape[0]

        # Insert bias units of 1 into the first column of the data matrix.
        data = np.insert(data, 0, 1, axis=1)

        # Calculate the activations and probabilities of the hidden units.
        hidden_activations = np.dot(data, self.weights)
        hidden_probs = self._logistic(hidden_activations)

        # Sample hidden states based on probabilities.
        hidden_states = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)

        # Ignore the bias units.
        return hidden_states[:, 1:]

    def run_hidden(self, data):
        """
        Run the network on a set of hidden units to get a sample of the visible units.

        Parameters
        ----------
        data: np.ndarray
            A matrix where each row consists of the states of the hidden units.

        Returns
        -------
        np.ndarray
            A matrix where each row consists of the visible units activated from the hidden units in the data matrix.
        """
        num_examples = data.shape[0]

        # Insert bias units of 1 into the first column of the data matrix.
        data = np.insert(data, 0, 1, axis=1)

        # Calculate the activations and probabilities of the visible units.
        visible_activations = np.dot(data, self.weights.T)
        visible_probs = self._logistic(visible_activations)

        # Sample visible states based on probabilities.
        visible_states = visible_probs > np.random.rand(num_examples, self.num_visible + 1)

        # Ignore the bias units.
        return visible_states[:, 1:]

    def daydream(self, num_samples):
        """
        Perform Gibbs sampling to generate visible unit samples from a randomly initialized state.

        Parameters
        ----------
        num_samples: int
            The number of samples to generate.

        Returns
        -------
        np.ndarray
            A matrix where each row is a sample of the visible units produced while the network was daydreaming.
        """
        # Initialize samples matrix with an extra bias unit.
        samples = np.ones((num_samples, self.num_visible + 1))

        # Take the first sample from a uniform distribution.
        samples[0, 1:] = np.random.rand(self.num_visible)

        # Perform Gibbs sampling.
        for i in range(1, num_samples):
            visible = samples[i-1, :]

            # Calculate hidden unit probabilities and states.
            hidden_activations = np.dot(visible, self.weights)
            hidden_probs = self._logistic(hidden_activations)
            hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
            hidden_states[0] = 1  # Fix the bias unit.

            # Calculate visible unit probabilities and states.
            visible_activations = np.dot(hidden_states, self.weights.T)
            visible_probs = self._logistic(visible_activations)
            visible_states = visible_probs > np.random.rand(self.num_visible + 1)
            samples[i, :] = visible_states

        # Ignore the bias units.
        return samples[:, 1:]

    def _logistic(self, x):
        """
        Compute the logistic sigmoid function.

        Parameters
        ----------
        x: np.ndarray
            The input array.

        Returns
        -------
        np.ndarray
            The output array after applying the logistic sigmoid function.
        """
        return 1.0 / (1 + np.exp(-x))
