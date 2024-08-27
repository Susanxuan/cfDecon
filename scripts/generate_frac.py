import numpy as np
import pickle


def generate_sample_array(n, c, sparse=True, sparse_prob=0.3, rare=False, rare_percentage=0.4):
    # Generate random values for each cell in the array
    values = np.random.rand(n, c)

    # Normalize the values to ensure the sum of each row is 1
    row_sums = np.sum(values, axis=1)
    prop = values / row_sums[:, np.newaxis]

    if sparse:
        print("You set sparse as True, some cell's fraction will be zero, the probability is", sparse_prob)
        ## Only partial simulated data is composed of sparse celltype distribution
        for i in range(int(prop.shape[0] * sparse_prob)):
            indices = np.random.choice(np.arange(prop.shape[1]), replace=False, size=int(prop.shape[1] * sparse_prob))
            prop[i, indices] = 0

        prop = prop / np.sum(prop, axis=1).reshape(-1, 1)

    if rare:
        print(
            'You will set some cell type fractions are very small (<3%), '
            'these celltype is randomly chosen by percentage you set before.')
        ## choose celltype
        np.random.seed(0)
        indices = np.random.choice(np.arange(prop.shape[1]), replace=False, size=int(prop.shape[1] * rare_percentage))
        prop = prop / np.sum(prop, axis=1).reshape(-1, 1)

        for i in range(int(0.5 * prop.shape[0]) + int(int(rare_percentage * 0.5 * prop.shape[0]))):
            prop[i, indices] = np.random.uniform(0, 0.03, len(indices))
            buf = prop[i, indices].copy()
            prop[i, indices] = 0
            prop[i] = (1 - np.sum(buf)) * prop[i] / np.sum(prop[i])
            prop[i, indices] = buf

    return prop


# Example usage
n = 1000  # Number of rows (samples)
c = 15  # Number of columns (cell types)
sparse = True
sparse_prob = 0.3
rare = False
rare_percentage = 0.4
sample_array = generate_sample_array(n, c, sparse=sparse, sparse_prob=sparse_prob, rare=rare, rare_percentage=rare_percentage)
print(sample_array)

filename = "./results/fractions/sample_array_" + str(n) + str(c) + "_spar" + str(sparse_prob) + str(rare) + str(
    rare_percentage) + ".pkl"
with open(filename, "wb") as file:
    pickle.dump(sample_array, file)
