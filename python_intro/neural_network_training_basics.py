import pandas as pd
import numpy as np

"""
x1 (GRE)   ─┐
x2 (GPA)   ─┼─► [ Σ w_i x_i ] ─► sigmoid ─► ŷ (合格確率)
x3 (rank1) ─┤
x4 (rank2) ─┤
x5 (rank3) ─┤
x6 (rank4) ─┘
"""

def sigmoid(x):
    return 1 / (1 + np.exp(-np.array(x, dtype=float)))

# derivative of sigmoid function
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def error_formula(y, output):
    return -y * np.log(output) - (1 - y) * np.log(1 - output)

def error_term_formula(y, output, output_grad):
    return (y - output) * output_grad

# Neural Network hyperparameters
epochs = 1000
learnrate = 0.5

# Training function
def train_nn(features, targets, epochs, learnrate):
    
    # Use to same seed to make debugging easier
    np.random.seed(42)

    # features (400x6)
    n_records, n_features = features.shape
    last_loss = None

    # Initialize weights (6,)の一次元ベクトル
    weights = np.random.normal(scale=1/n_features**.5, size=n_features)
    print(weights)

    for e in range(epochs):
        del_w = np.zeros(weights.shape)

        # x: featuresは360x6で、特徴量ベクトル[gre, gpa, rank_1, rank_2, rank_3, rank_4] (6,)の一次元ベクトル
        # y: targetsは360x1で、1か0で、1は合格、0は不合格 (実際は(360,)の一次元ベクトル)
        for x, y in zip(features.values, targets):
            # Loop through all records, x is the input, y is the target

            # Activation of the output unit
            #   Notice we multiply the inputs and the weights here 
            #   rather than storing h as a separate variable 
            output = sigmoid(np.dot(x, weights))

            # The error, the target minus the network output
            error = error_formula(y, output)

            # The error term
            output_grad = sigmoid_prime(output)
            error_term = error_term_formula(y, output, output_grad)

            # The gradient descent step, the error times the gradient times the inputs
            del_w = np.add(del_w, error_term * x)

        # Update the weights here. The learning rate times the 
        # change in weights, divided by the number of records to average
        weights = np.add(weights, learnrate * del_w / n_records)

        # Printing out the log-loss error on the training set
        if e % (epochs / 10) == 0:
            out = sigmoid(np.dot(features, weights))

            # We are using binary cross-entropy (log-loss) to monitor training progress 
            # as well as for computing gradients, to stay consistent with the loss function 
            # used in training.
            loss = np.mean(error_formula(targets, out))
            print("Epoch:", e)
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            print("=========")
    print("Finished training!")
    return weights

data = pd.read_csv("student_data.csv")
print(data[:10])

# one-hot encode the rank
one_hot_data = pd.concat([data, pd.get_dummies(data['rank'], prefix='rank')], axis=1)
print(one_hot_data[:10])

one_hot_data = one_hot_data.drop('rank', axis=1)
print(one_hot_data[:10])

# scale the column
processed_data = one_hot_data[:]
processed_data["gre"] = processed_data["gre"] / 800.0
processed_data["gpa"] = processed_data["gpa"] / 4.0
print(processed_data)

# split the data into training and test sets
sample = np.random.choice(processed_data.index, size=int(len(processed_data)*0.9), replace=False)
train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)

# split the data into features (X) and targets (y)
features = train_data.drop("admit", axis=1)
targets = train_data["admit"]
features_test = test_data.drop("admit", axis=1)
targets_test = test_data["admit"]
print(features[:10])
print(targets[:10])

# training
weights = train_nn(features, targets, epochs, learnrate)

# Calculate accuracy on test data
test_out = sigmoid(np.dot(features_test, weights))
predictions = test_out > 0.5
accuracy = np.mean(predictions == targets_test)
print(f"Prediction accuracy: {accuracy:.3f}")