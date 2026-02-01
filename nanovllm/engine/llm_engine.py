import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        # 读取配置
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        # 为每个GPU创建一个worker进程，负责计算
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn") # 使用spawn创建进程，而非fork。spawn创建的进程是完整独立的进程，fork进程的内存依赖copy-on-write，无独立的内存栈，CUDA不安全
        for i in range(1, config.tensor_parallel_size): # 创建-1个worker进程
            event = ctx.Event() # 进程间同步
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # 创建master进程，负责调度，分发任务到worker，计算
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True) # 转换字符和token
        config.eos = self.tokenizer.eos_token_id # 输出停止的token ID
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    # 优雅退出
    def exit(self):
        self.model_runner.call("exit") # master执行退出前的工作
        del self.model_runner
        for p in self.ps:
            p.join() # 等待worker进程完成计算后退出

    # 接收请求
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt) # 将input转为token
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq) # 将input添加到调度器中

    # 执行一次调度
    def step(self):
        seqs, is_prefill = self.scheduler.schedule() # 从调度器中获取待处理的请求。prefill与decode分离，可以高效利用GPU资源
        token_ids = self.model_runner.call("run", seqs, is_prefill) # 执行批量推理，获取输出的token ids
        self.scheduler.postprocess(seqs, token_ids) # 后处理：从队列中移除已经完成的请求，释放相关资源
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs) # 统计本次调度处理的token：prefill阶段处理所有token，decode阶段处理一个token
        return outputs, num_tokens

    # 无running、waiting的请求，调度器状态为finished才能退出进程
    def is_finished(self):
        return self.scheduler.is_finished()

    # 用户调用的入口
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm: # 打印进度条
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp) # 添加请求到调度器
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step() # 执行一次调度
            if use_tqdm:
                if num_tokens > 0: # 统计tgs
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs] # decode
        if use_tqdm:
            pbar.close()
        return outputs
