from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs # 批处理上限
        self.max_num_batched_tokens = config.max_num_batched_tokens # 批处理token上限
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque() # waiting队列，新进入的seq，需要prefill
        self.running: deque[Sequence] = deque() # running队列，已经prefill的seq，需要decode

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq) # 新请求添加到waiting队列中

    # 执行一次调度
    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = [] # 本次调度的seq
        num_seqs = 0 # 本次调度的seq数量+正在运行的seq数量
        num_batched_tokens = 0 # 正在运行的seq所占用的token
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0] # 获取waiting队首的seq
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break # 批处理token超限，或者空闲显存不足以处理该请求，则结束此次调度
            num_seqs += 1
            self.block_manager.allocate(seq) # 为seq申请显存
            num_batched_tokens += len(seq) - seq.num_cached_tokens # 统计seq所消耗的token，非首次处理的seq，需要减去上一次cache的token
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft() # 符合调度条件，waiting队首的seq出队
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs: # waiting中是新进入的seq，需要prefill
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq): # 显存不足以处理该seq时，通过抢占式调度该seq，抢占式调度保证公平性，牺牲了运行中的seq的响应时间，为了及时响应新进来的seq
                if self.running: # 释放running队列中最后一个seq的显存，并撤回到waiting队列
                    self.preempt(self.running.pop())
                else: # running队列所有seq撤回waiting并释放显存后，还不足以处理该seq，则将该seq撤回到waiting队列
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq) # 为该seq预留显存
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False # running队列中的seq是已经prefill完，需要decode的seq

    # 抢占式调度，将已经prefill的seq撤回到waiting队首，并释放占用的显存
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    # 释放已经处理完成的seq占用的显存
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
