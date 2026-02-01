import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    """
    model: 模型文件路径或模型ID
    max_num_batched_tokens: 每个batch所能处理的最大token数，用于 
        1.防止显存溢出
        2.平衡latency和throughput（更大的batch意味着更大的throughput，但是latency也会增大）
    max_num_seqs: 每个batch所能处理的最大请求数，调大可以提升throughput，需要更多显存，导致更大的latency
    max_model_len: 每个请求所能处理的最大token数，即input+output的最大值，配置max_model_len时不能超过模型的max_position_embeddings
    gpu_memory_utilization: vllm能使用的最大显存比例。假如为0.9，则
        90%的显存用于存储 
            1.模型权重
            2.KV Cache（受seq_len和num_seq影响）
            3.前向计算时产生的临时数据
        10%的显存用于存储
            1.PyTorch/CUDA操作
            2.其他GPU操作
    tensor_parallel_size: 将模型权重拆分到多张卡上，
        拆法为：将每个layer拆成tensor_parallel_size份。
        一个layer层的计算需要多张GPU协作完成
            1  2   *   5  6   =   19  22   
            3  4       7  8       43  50
        拆分权重到两卡上
            1  2   *   5 =   19 
            3  4       7     43   
                                    --->   19  22
            1  2   *   6 =   22            43  50
            3  4       8     50
        与之对应的是pipeline并行，将多层layer分配到多张卡上，GPU之间传递的是计算结果
    enforce_eager: 使用eager或CUDA graph推理，
        eager：
            CPU：Launch Kernel 1 (Add)
            CPU：wait
            CPU：Launch Kernel 2 (ReLU)
        CUDA graph：
            Phase 1: Graph Capture (once)
                graph.record_start()
                → Kernel 1: Add
                → Kernel 2: ReLU
                graph.record_end()
            Phase 2: Graph Replay (every time after)
                CPU：Launch Graph (ADD， ReLU)
        与Fusion Kernel的区别
            Fusion Kernel：优化算子本身；实现复杂，需要重写算子
            CUDA graph：减少launch kernel的次数，launch kernel的开销很大；实现简单，只需要record即可
    hf_config：模型配置文件
    eos: 结束符号的token id。模型停止输出的条件之一
        1.模型输出eos
        2.input+output超过max_model_len
    kvcache_block_size：每个memory block存储多少个token的kv，为了对齐内存，最好设置为256的倍数。例如一个block存储16个kv，那么50个token的seq会申请3*16+1*2=4个block。block的好处
        1.按需申请，无需预申请
        2.减少内存碎片
    num_kvcache_blocks: block的数量，影响最大并发数。例如一个seq需要10个block，那么最大并发数为num_kvcache_blocks / 10
    """
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
