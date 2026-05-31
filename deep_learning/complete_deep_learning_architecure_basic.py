import numpy as np
from data_prep import features, targets, features_test, targets_test

features = features.to_numpy(dtype=np.float64)
targets = targets.to_numpy(dtype=np.float64)
features_test = features_test.to_numpy(dtype=np.float64)
targets_test = targets_test.to_numpy(dtype=np.float64)

"""
binary.csvの全データ数: 400行
そのうち90%を学習用、10%をテスト用
特徴量: gre, gpa, rank_1, rank_2, rank_3, rank_4

features: (360, 6) -> 360サンプル、各サンプルは6次元ベクトル
targets: (360,) -> 中身合格かどうか(0か1)

features_test: (40, 6)
targets_test: (40,)

ロジスティック回帰 (Logistic regression)
z = w1x1 + w2x2 + ... + w6x6 + b
logit ŷ = sigmoid(z)

ロジスティック回帰 = 単層NN
6次元ロジスティック回帰 * 2 -> 2次元ロジスティック回帰

6次元入力を持つ2層ニューラルネットワーク
入力層 (6)    隠れ層 (2)    出力層 (1)
gre     ───────▶ h1 ──┐
gpa     ───────▶ h1   │
rank_1  ───────▶ h1   │
rank_2  ───────▶ h2 ──┼──▶ admit (ŷ)
rank_3  ───────▶ h2   │
rank_4  ───────▶ h2 ──┘
"""

def sigmoid(x):
    """
    Calculate sigmoid
    """
    x = np.asarray(x, dtype=np.float64)
    return 1 / (1 + np.exp(-x))

def forward_pass(x, weights_input_to_hidden, weights_hidden_to_output):
    """
    Make a forward pass through the network
    """
    # Calculate the input to the hidden layer.
    hidden_layer_in = np.dot(x, weights_input_to_hidden)
    # Calculate the hidden layer output.
    hidden_layer_out = sigmoid(hidden_layer_in)

    # Calculate the input to the output layer.
    output_layer_in = np.dot(hidden_layer_out, weights_hidden_to_output)
    # Calculate the output of the network.
    output_layer_out = sigmoid(output_layer_in)

    return hidden_layer_out, output_layer_out


"""
バックプロパゲーションの基本形
Δw = 学習率 ×（その重みに流れ込む入力）×（その層の error term）
"""
def backward_pass(x, target, learnrate, hidden_layer_out, output_layer_out, weights_hidden_to_output):
    """
    Make a backward pass through the network
    """
    # Calculate output error
    error = target - output_layer_out

    # Calculate error term for output layer
    output_error_term = error * output_layer_out * (1 - output_layer_out)

    # Calculate error term for hidden layer
    hidden_error_term = np.dot(output_error_term, weights_hidden_to_output) * hidden_layer_out * (1 - hidden_layer_out)

    # Calculate change in weights for hidden layer to output layer
    delta_w_h_o = learnrate * output_error_term * hidden_layer_out

    # Calculate change in weights for input layer to hidden layer
    delta_w_i_h = learnrate * hidden_error_term * x[:, None]

    return delta_w_h_o, delta_w_i_h

def update_weights(weights_input_to_hidden, weights_hidden_to_output, 
                   features, targets, learnrate):
    """
    Complete a single epoch of gradient descent and return updated weights
    """
    # weights_input_to_hidden = (6, 2) 
    delta_w_i_h = np.zeros(weights_input_to_hidden.shape)
    # weights_hidden_to_output = (2,)
    delta_w_h_o = np.zeros(weights_hidden_to_output.shape)
    
    # Loop through all records, x is the input, y is the target
    for x, y in zip(features, targets):
        ## Forward pass ##
        
        # Calculate the output using the forward_pass function.
        hidden_layer_out, output_layer_out = forward_pass(x, weights_input_to_hidden, weights_hidden_to_output)
        
        ## Backward pass ##
        
        # Calculate the change in weights using the backward_pass　function.
        delta_w_h_o_temp, delta_w_i_h_temp = backward_pass(x, y, learnrate, hidden_layer_out, output_layer_out, weights_hidden_to_output)
        delta_w_h_o += delta_w_h_o_temp
        delta_w_i_h += delta_w_i_h_temp

    n_records = features.shape[0]
    # Update weights  (division by n_records or number of samples). 
    weights_input_to_hidden +=  delta_w_i_h / n_records
    weights_hidden_to_output +=  delta_w_h_o / n_records
    
    return weights_input_to_hidden, weights_hidden_to_output

def gradient_descent(features, targets, epochs=2000, learnrate=0.9):
    """
    Perform the complete gradient descent process on a given dataset
    """
    # Use to same seed to make debugging easier
    np.random.seed(11)
    
    # Initialize loss and weights
    last_loss = None
    n_features = features.shape[1]
    n_hidden = 2

    # 重みの初期化、正規分布から乱数を生成している
    # scale = 1/sqrt(n_features)は標準偏差(1/sqrt(6) = 0.408) -> ほとんど(約68%)の重みは[-0.4, +0.4]に収まるという意味
    # 「入力がn本あるなら、1本あたりの影響を1/√nに薄める」
    weights_input_hidden = np.random.normal(scale=1 / n_features ** .5, size=(n_features, n_hidden))
    weights_hidden_output = np.random.normal(scale=1 / n_features ** .5, size=n_hidden)

    # Repeatedly update the weights based on the number of epochs
    for e in range(epochs):
        weights_input_hidden, weights_hidden_output = update_weights(
            weights_input_hidden,
            weights_hidden_output,
            features,
            targets,
            learnrate
        )

        # Printing out the MSE on the training set every 10 epochs.
        if e % (epochs / 10) == 0:
            hidden_output = sigmoid(np.dot(features, weights_input_hidden))
            out = sigmoid(np.dot(hidden_output, weights_hidden_output))

            # MSE: Mean Squared Errorで予測値と実際の値の差(誤差)の2乗の平均を計算する損失関数
            loss = np.mean((out - targets) ** 2)
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            
    return weights_input_hidden, weights_hidden_output

def calculate_accuracy(features, targets, weights_input_hidden, weights_hidden_output):
    """
    Given features, targets, and weights for both hidden and output
    layers, calculate the accuracy of predictions
    """
    hidden_out = sigmoid(np.dot(features, weights_input_hidden))
    output_out = sigmoid(np.dot(hidden_out, weights_hidden_output))
    predictions = output_out > 0.5
    accuracy = np.mean(predictions == targets)
    return accuracy

# Calculate accuracy on test data
weights_input_hidden, weights_hidden_output = gradient_descent(features, targets)
accuracy = calculate_accuracy(features_test, targets_test, weights_input_hidden, weights_hidden_output)
print("Prediction accuracy: {:.3f}".format(accuracy))
