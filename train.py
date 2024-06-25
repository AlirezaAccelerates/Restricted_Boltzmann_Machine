import numpy as np
from RBM import RBM

def create_dataset(num_examples, num_visible):
    """
    Create a synthetic dataset for demonstration purposes.

    Parameters
    ----------
    num_examples: int
        Number of examples in the dataset.
    num_visible: int
        Number of visible units.

    Returns
    -------
    np.ndarray
        A matrix where each row is a training example consisting of random binary states of visible units.
    """
    return np.random.rand(num_examples, num_visible)


if __name__ == "__main__":
    # Parameters
    num_visible = 150
    num_hidden = 5
    num_examples = 10
    max_epochs = 5000
    learning_rate = 0.1

    # Create a synthetic dataset
    data = create_dataset(num_examples, num_visible)

    # Initialize the RBM
    rbm = RBM(num_visible=num_visible, num_hidden=num_hidden)

    # Train the RBM
    print("Training RBM...")
    final_error = rbm.train(data, max_epochs=max_epochs, learning_rate=learning_rate)
    print(f"Final reconstruction error: {final_error}")

    # Run the RBM on the visible units to get hidden states
    hidden_states = rbm.run_visible(data)
    print("Hidden states:\n", hidden_states)

    # Run the RBM on the hidden units to get visible states
    visible_states = rbm.run_hidden(hidden_states)
    print("Reconstructed visible states:\n", visible_states)

    # Generate samples using the trained RBM
    num_samples = 10
    samples = rbm.daydream(num_samples=num_samples)
    print("Generated samples:\n", samples)
