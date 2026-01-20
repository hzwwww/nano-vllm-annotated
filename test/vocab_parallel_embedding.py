import torch
import torch.nn as nn
import torch.nn.functional as F

# 词汇表定义
VOCAB = [
    "a", "big", "cat", "sits", "on", "the",   # 这 6 个词属于 Rank 0
    "mat", "and", "a", "small", "dog", "plays" # 这 6 个词属于 Rank 1
]
# 注意：为了演示方便，"a" 在词汇表里出现了两次，分别在两个分片中。
# 在真实场景中，词汇表中的词是唯一的。

word_to_id = {word: i for i, word in enumerate(VOCAB)}
id_to_word = {i: word for i, word in enumerate(VOCAB)}

# VocabParallelEmbedding 类保持不变，它只处理数字ID
class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tp_rank: int,
        tp_size: int,
    ):
        super().__init__()
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))

    def forward(self, x: torch.Tensor):
        mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
        print(f"\n[步骤 1: 生成的掩码 (Mask)]")
        print(mask)

        local_x = (x - self.vocab_start_idx) * mask
        print(f"\n[步骤 2: 转换后的本地索引]")
        print(local_x)

        y = F.embedding(local_x, self.weight)
        print(f"\n[步骤 3: 本地 Embedding 查找结果]")
        print(y)
        
        y = y * mask.unsqueeze(-1)
        print(f"\n[步骤 4: 清零后的输出 (准备 All-Reduce)]")
        print(y)
        
        return y

# --- 模拟环境设置 ---
VOCAB_SIZE = len(VOCAB)
EMBEDDING_DIM = 4
TP_SIZE = 2

# 输入从ID改为文字
text_input = ["cat", "dog"]
input_ids = torch.tensor([[word_to_id[word] for word in text_input]], dtype=torch.long)


print("="*60)
print(f"模拟开始: TP_SIZE={TP_SIZE}, VOCAB_SIZE={VOCAB_SIZE}")
print(f"词汇表: {VOCAB}")
print(f"输入文字: {text_input}")
print(f"转换后的输入 Token IDs: {input_ids.flatten().tolist()}")
print("="*60)

# --- 手动模拟每个 Rank 的执行过程 ---
partial_outputs = []
all_embeddings_for_verification = []

for tp_rank in range(TP_SIZE):
    print(f"\n{'#'*20} 模拟 Rank {tp_rank} 的计算过程 {'#'*20}")
    
    embedding_layer = VocabParallelEmbedding(VOCAB_SIZE, EMBEDDING_DIM, tp_rank, TP_SIZE)
    partition_size = VOCAB_SIZE // TP_SIZE
    
    # 打印当前GPU负责的词汇
    start_idx = embedding_layer.vocab_start_idx
    end_idx = embedding_layer.vocab_end_idx
    vocab_slice = [f"'{id_to_word[i]}':{i}" for i in range(start_idx, end_idx)]
    print(f"\nRank {tp_rank} 负责的词汇 (及其全局ID): \n{vocab_slice}")
    
    # 设置权重
    weights_data = torch.arange(
        start_idx, end_idx, dtype=torch.float32
    ).unsqueeze(1) * 10.0 + torch.arange(EMBEDDING_DIM, dtype=torch.float32)
    
    with torch.no_grad():
        embedding_layer.weight.copy_(weights_data)
        all_embeddings_for_verification.append(weights_data)

    print(f"\nRank {tp_rank} 的权重分片:")
    print(embedding_layer.weight)

    partial_y = embedding_layer(input_ids)
    partial_outputs.append(partial_y)

# --- 模拟 All-Reduce 操作 ---
print(f"\n\n{'='*25} 模拟 All-Reduce 聚合 {'='*25}")
for i, p_out in enumerate(partial_outputs):
    print(f"\n来自 Rank {i} 的贡献:")
    print(p_out)

final_output = torch.stack(partial_outputs).sum(dim=0)
print(f"\n聚合后的最终结果:")
print(final_output)

# --- 验证结果 ---
print(f"\n\n{'='*28} 验证结果 {'='*28}")
full_weight_matrix = torch.cat(all_embeddings_for_verification, dim=0)
print(f"\n完整权重矩阵 (每一行代表一个单词的向量):")
print(full_weight_matrix)

standard_embedding_layer = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
with torch.no_grad():
    standard_embedding_layer.weight.copy_(full_weight_matrix)
standard_output = standard_embedding_layer(input_ids)
print(f"\n标准 nn.Embedding 计算结果:")
print(standard_output)

# 更加直观地对比单个词的结果
word_to_check = "cat"
word_id = word_to_id[word_to_check]
word_index_in_batch = text_input.index(word_to_check)

print(f"\n--- 单独验证单词 '{word_to_check}' (ID: {word_id}) 的向量 ---")
print(f"并行计算得到的向量:\n{final_output[0, word_index_in_batch]}")
print(f"标准计算得到的向量:\n{standard_output[0, word_index_in_batch]}")

are_equal = torch.allclose(final_output, standard_output)
print(f"\n并行计算结果与标准计算结果是否一致: {are_equal}")