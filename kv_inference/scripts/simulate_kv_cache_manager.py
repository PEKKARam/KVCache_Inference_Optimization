from kv_inference.src.kvcache_manager import PagedKVCacheAllocator, ContinuousBatcher, Request


def run_allocator_demo():
    allocator = PagedKVCacheAllocator(block_size=16, total_blocks=64)
    allocator.allocate(seq_id=1, num_tokens=20)
    allocator.allocate(seq_id=2, num_tokens=45)
    allocator.allocate(seq_id=3, num_tokens=10)

    print("Allocated blocks:", allocator.allocated_blocks())
    print("Fragmentation tokens:", allocator.fragmentation())

    allocator.free_sequence(2)
    print("Allocated blocks after free:", allocator.allocated_blocks())


def run_continuous_batching_demo():
    batcher = ContinuousBatcher(max_batch_size=2)
    batcher.add(Request(request_id=1, prompt_tokens=8, gen_tokens=8))
    batcher.add(Request(request_id=2, prompt_tokens=16, gen_tokens=4))
    batcher.add(Request(request_id=3, prompt_tokens=4, gen_tokens=12))

    step = 0
    while batcher.has_work():
        finished = batcher.step()
        active_ids = [r.request_id for r in batcher.active]
        print(f"step {step}: active={active_ids}, finished={[r.request_id for r in finished]}")
        step += 1


if __name__ == "__main__":
    run_allocator_demo()
    run_continuous_batching_demo()
