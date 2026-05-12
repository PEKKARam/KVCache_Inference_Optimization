from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class KVBlock:
    block_id: int
    capacity: int
    used: int = 0

    def remaining(self):
        return self.capacity - self.used


class PagedKVCacheAllocator:
    def __init__(self, block_size, total_blocks):
        self.block_size = block_size
        self.total_blocks = total_blocks
        self.free_blocks = list(range(total_blocks))
        self.sequence_blocks: Dict[int, List[KVBlock]] = {}

    def allocate(self, seq_id, num_tokens):
        blocks = self.sequence_blocks.get(seq_id, [])
        while num_tokens > 0:
            if blocks and blocks[-1].remaining() > 0:
                take = min(num_tokens, blocks[-1].remaining())
                blocks[-1].used += take
                num_tokens -= take
                continue
            if not self.free_blocks:
                raise RuntimeError("Out of KV blocks")
            block_id = self.free_blocks.pop(0)
            new_block = KVBlock(block_id=block_id, capacity=self.block_size, used=0)
            blocks.append(new_block)
        self.sequence_blocks[seq_id] = blocks

    def free_sequence(self, seq_id):
        blocks = self.sequence_blocks.pop(seq_id, [])
        for block in blocks:
            block.used = 0
            self.free_blocks.append(block.block_id)

    def allocated_blocks(self):
        return self.total_blocks - len(self.free_blocks)

    def fragmentation(self):
        wasted = 0
        for blocks in self.sequence_blocks.values():
            if blocks:
                wasted += blocks[-1].remaining()
        return wasted


@dataclass
class Request:
    request_id: int
    prompt_tokens: int
    gen_tokens: int
    position: int = 0

    def step(self):
        if self.position < self.prompt_tokens + self.gen_tokens:
            self.position += 1
            return True
        return False

    def is_done(self):
        return self.position >= self.prompt_tokens + self.gen_tokens


class ContinuousBatcher:
    def __init__(self, max_batch_size):
        self.max_batch_size = max_batch_size
        self.active: List[Request] = []
        self.pending: List[Request] = []

    def add(self, request: Request):
        self.pending.append(request)

    def step(self):
        while len(self.active) < self.max_batch_size and self.pending:
            self.active.append(self.pending.pop(0))
        finished = []
        for req in self.active:
            req.step()
            if req.is_done():
                finished.append(req)
        self.active = [r for r in self.active if r not in finished]
        return finished

    def has_work(self):
        return bool(self.active) or bool(self.pending)
