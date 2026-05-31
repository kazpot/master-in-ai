import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, RandomSampler

"""
このコードは、PyTorchを用いて文字レベルの言語モデルをゼロから実装・学習するデモです。

主な特徴:
- tiny-shakespeare データセットを用いた文字単位（character-level）の学習
- Transformer Decoder（GPT型）アーキテクチャを採用
- Causal Self-Attention により「過去のトークンのみ」を参照する自己回帰モデル
- Token Embedding + Positional Embedding
- Multi-Head Self-Attention、Feed Forward Network、Residual Connection、LayerNormを備えた典型的なTransformer Block
- 各位置で次の文字を予測するNext-Token Predictionにより学習
- 学習後は 1 トークンずつ自己回帰的に文章生成が可能

本コードは事前学習済みモデルを利用しないスクラッチ実装であり、Transformer / GPT の構造理解を目的とした教育・検証用の実装である。
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

text = Path("tiny-shakespeare.txt").read_text()
print(text[0:1000])

class CharTokenizer:
    def __init__(self, vocabulary):
        self.token_id_for_char = {char: token_id for token_id, char in enumerate(vocabulary)}
        self.char_for_token_id = {token_id: char for token_id, char in enumerate(vocabulary)}
    
    @staticmethod
    def train_from_text(text):
        vocabulary = set(text)
        return CharTokenizer(sorted(list(vocabulary)))
    
    def encode(self, text):
        token_ids = []
        for char in text:
            token_ids.append(self.token_id_for_char[char])
        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, token_ids):
        chars = []
        for token_id in token_ids.tolist():
            chars.append(self.char_for_token_id[token_id])
        return "".join(chars)

    def vocabulary_size(self):
        return len(self.token_id_for_char)

class TokenIdsDataset(Dataset):
    # block_size: 学習用データとして切り出す文字列（入力）の長さ
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, pos):
        x = self.data[pos:pos + self.block_size]
        y = self.data[pos + 1: pos + 1 + self.block_size]
        return x, y


tokenizer = CharTokenizer.train_from_text(text)
print(tokenizer.encode("Hello world"))
print(tokenizer.decode(tokenizer.encode("Hello world")))
print(tokenizer.vocabulary_size())

tokenizedText = tokenizer.encode(text)
dataset = TokenIdsDataset(tokenizedText, block_size=64)

x, y = dataset[0]
print(x)
print(tokenizer.decode(x))

sampler = RandomSampler(dataset, replacement=True)
dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)
x, y = next(iter(dataloader))
print(x.shape)
print(x)
print(tokenizer.decode(x[0]))
print(tokenizer.decode(y[0]))

# Wrod Embeddings
# Large -> tokenID 348 -> [-1.00, 0.58, ..., 0.43]
# models -> tokenID 634 -> [0.25, 0.04, ..., -1.14]

vocab_size = 10000
embedding_dim = 75
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
token_ids = torch.LongTensor([11, 9783, 376, 45])
embedded = embedding(token_ids)
print(embedded.shape)

# Positional Encoding: 
# Injects positional information into the word embeddings, 
# helping the model understand the relative order of input tokens.

# Word Embeddings + Positional Encoding = Transformer model

# Softmax Function
# create a probability distribution from logits
# two conditions must be met:
# 1. Every logit value must be converted into a value between 0 and 1.
# 2. The sum of all resulting values must equal one.
# neural network -> [+1.2 +1.41 ... -2.1] ->  Softmax -> [e^1.2 / sum(e^n) e^1.41 / sum(e^n) ... ]

# logit（ロジット）＝ ニューラルネットワークが「生のまま」出力するスコア -> 「どれが有力か」を表す 相対的な強さ
# softmax ＝ そのスコアを「確率」に変換する関数

# Calculating Attention
# Attention Block: Computes attention scores and updates embedding values using these scores.

# 1. Queries Matrix
# Multiply the matrix of embedding vectors by 
# another matrix ( W_Q ) to produce the query matrix (Q)

# 2. Key Matrix
# the key matrix (K) is obtained b
# y multiplying the embedding vectors by matrix ( W_K )

# 3. Computing Attention Scores
# Calculate the dot product between vectors in the Q and K matrices
# QK^T = Attention scores

# 4. Normalization with Softmax
# Apply the softmax function to each column of the resulting matrix.

# 5. Value Matrix (V)
# Similar to Q and K, the value matrix (V) is derived using another matrix ( W_V ).
# Multiply each vector in V by the corresponding attention scores 
# to produce updated embeddings.

# Attention(Q, K, V) = softmax(QK^T / sqrt(d))V

# 6. Masking in the Attention Matrix
# Assign all values below a certain diagonal to negative infinity.

# implementing attention block

config = {
    "vocabulary_size": tokenizer.vocabulary_size(),
    "context_size": 256,
    "embedding_dim": 768,
    "heads_num": 12,
    "layers_num": 10,
    "dropout_rate": 0.1,
    "use_bias": False,
}

config["head_size"] = config["embedding_dim"] // config["heads_num"]

class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Q_weights = nn.Linear(config["embedding_dim"], config["head_size"], config["use_bias"])
        self.K_weights = nn.Linear(config["embedding_dim"], config["head_size"], config["use_bias"])
        self.V_weights = nn.Linear(config["embedding_dim"], config["head_size"], config["use_bias"])

        self.dropout = nn.Dropout(config["dropout_rate"])

        casual_attention_mask = torch.tril(torch.ones(config["context_size"], config["context_size"]))
        self.register_buffer("casual_attention_mask", casual_attention_mask)

    def forward(self, input): # (B, C, embedding_dim)
        batch_size, token_num, embedding_dim = input.shape
        Q = self.Q_weights(input) # (B, C, head_size)
        K = self.K_weights(input) # (B, C, head_size)
        V = self.V_weights(input) # (B, C, head_size)

        attention_scores = Q @ K.transpose(1, 2) # (B, C, C)
        attention_scores = attention_scores.masked_fill(self.casual_attention_mask[:token_num,:token_num] == 0, -torch.inf)
        attention_scores = attention_scores / (K.shape[-1] ** 0.5)
        attention_scores = torch.softmax(attention_scores, dim=1)
        attention_scores = self.dropout(attention_scores)

        return attention_scores @ V # (B, C, head_size)

input = torch.rand(8, config["context_size"], config["embedding_dim"])
ah = AttentionHead(config)
output = ah(input)
print(output.shape)

# 同じ入力を複数の独立した attention で並列に解析し、それらを統合して、より豊かな表現を得る仕組み
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        heads_list = [AttentionHead(config) for _ in range(config["heads_num"])]
        self.heads = nn.ModuleList(heads_list)
        self.linear = nn.Linear(config["embedding_dim"], config["embedding_dim"])
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, input):
        heads_output = [head(input) for head in self.heads]
        scores_change = torch.cat(heads_output, dim=-1)
        scores_change = self.linear(scores_change)
        return self.dropout(scores_change)
    
mha = MultiHeadAttention(config)
output = mha(input)
print(output.shape)

# Transformer Block: Each input token is converted into an embedding vector and passed to the multi-head attention block.
# GELU Activation Function: GELU is similar to the ReLU activation function but has a smoother shape, which helps in achieving better training results.
# Residual Connections: Helps with the vanishing gradient problem in deep networks.
# Layer Normalization Layer: Scales the output so that the mean becomes zero and variance becomes one, leading to faster training.

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(config["embedding_dim"], config["embedding_dim"] * 4),
            nn.GELU(),
            nn.Linear(config["embedding_dim"] * 4, config["embedding_dim"]),
            nn.Dropout(config["dropout_rate"])
        )

    def forward(self, input):
        return self.linear_layers(input)


ff = FeedForward(config)
input = torch.rand(8, config["context_size"], config["embedding_dim"])
output = ff(input)
print(output.shape)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.multi_head = MultiHeadAttention(config)
        self.layer_norm_1 = nn.LayerNorm(config["embedding_dim"])
        self.feed_forward = FeedForward(config)
        self.layer_norm_2 = nn.LayerNorm(config["embedding_dim"])

    def forward(self, input):
        residual = input
        x = self.multi_head(self.layer_norm_1(input))
        x = x + residual
        residual = x
        x = self.feed_forward(self.layer_norm_2(x))
        return x + residual
    
tb = TransformerBlock(config)
input = torch.randn(8, config["context_size"], config["embedding_dim"])
output = tb(input)
print(output.shape)

class DemoGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding_layer = nn.Embedding(config["vocabulary_size"], config["embedding_dim"])
        self.positional_embedding_layer = nn.Embedding(config["context_size"], config["embedding_dim"])
        blocks = [TransformerBlock(config) for _ in range(config["layers_num"])]
        self.layers = nn.Sequential(*blocks)
        self.layer_norm = nn.LayerNorm(config["embedding_dim"])
        self.unembedding = nn.Linear(config["embedding_dim"], config["vocabulary_size"], bias=False)

    def forward(self, token_ids):
        batch_size, tokens_num = token_ids.shape
        x = self.token_embedding_layer(token_ids)
        sequence = torch.arange(tokens_num, device=device)
        x = x + self.positional_embedding_layer(sequence)
        x = self.layers(x)
        x = self.layer_norm(x)
        x = self.unembedding(x)
        return x
    
model = DemoGPT(config).to(device)
output = model(tokenizer.encode("Hi").unsqueeze(dim=0).to(device))
print(output.shape)

def generate(model, prompt_ids, max_tokens):
    output_ids = prompt_ids
    for _ in range(max_tokens):
        if output_ids.shape[1] >= config["context_size"]:
            break
        with torch.no_grad():
            logits = model(output_ids)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        output_ids = torch.cat([output_ids, next_token_id], dim=1)
    return output_ids

def generate_with_prompt(model, tokenizer, prompt, max_tokens=100):
    model.eval()
    prompt = tokenizer.encode(prompt).unsqueeze(dim=0).to(device)
    return tokenizer.decode(generate(model, prompt, max_tokens=max_tokens)[0])

print(generate_with_prompt(model, tokenizer, "First Citizen:\n"))

# Training Setup
batch_size = 64
train_iterations = 5000
evaluation_interval = 100
learning_rate = 4e-4
train_split = 0.9

tokenized_text = tokenizer.encode(text).to(device)
train_count = int(train_split * len(tokenized_text))
train_data, validation_data = tokenized_text[:train_count], tokenized_text[train_count:]

train_dataset = TokenIdsDataset(train_data, config["context_size"])
validation_dataset = TokenIdsDataset(validation_data, config["context_size"])

train_sampler = RandomSampler(train_dataset, num_samples=batch_size * train_iterations, replacement=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

validation_sampler = RandomSampler(validation_dataset, replacement=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, sampler=validation_sampler)

@torch.no_grad()
def calculate_validation_loss(model, batches_num):
    model.eval()
    total_loss = 0
    validation_iter = iter(validation_dataloader)
    
    for _ in range(batches_num):
        input, targets = next(validation_iter)
        logits = model(input)
        logits_view = logits.view(batch_size * config["context_size"], config["vocabulary_size"])
        targets_view = targets.view(batch_size * config["context_size"])
        loss = F.cross_entropy(logits_view, targets_view)
        total_loss += loss.item()
    
    average_loss = total_loss / batches_num
    return average_loss

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step_num, sample in enumerate(train_dataloader):
  model.train()
  input, targets = sample
  logits = model(input)

  logits_view = logits.view(batch_size * config["context_size"], config["vocabulary_size"])
  targets_view = targets.view(batch_size * config["context_size"])
  
  loss = F.cross_entropy(logits_view, targets_view)
  # Backward propagation
  loss.backward()
  # Update model parameters
  optimizer.step()
  # Set to None to reduce memory usage
  optimizer.zero_grad(set_to_none=True)

  print(f"Step {step_num}. Loss {loss.item():.3f}")

  if step_num % evaluation_interval == 0:
    print("Demo GPT:\n" + generate_with_prompt(model, tokenizer, "\n"))
    validation_loss = calculate_validation_loss(model, batches_num=10)
    print(f"Step {step_num}. Validation loss: {validation_loss:.3f}")