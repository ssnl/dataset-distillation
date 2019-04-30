import torch
import torch.distributed as dist


if not dist.is_available():
    # distributed is only supported on Linux, make the following dummy impls for
    # running on macOS & Windows.

    def dummy(*args, **kwargs):
        raise RuntimeError(
            "Trying to run distributed code while torch.distributed is not available. "
            "Make sure that you are on Linux.")

    broadcast_coalesced = all_reduce_coalesced = all_gather_coalesced = barrier = dummy

else:

    try:
        from torch.distributed import ReduceOp
    except ImportError:
        from torch.distributed import reduce_op
        ReduceOp = reduce_op

    # NB: this might be broken by a future pytorch release
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, \
        _take_tensors

    MB = 1024 * 1024

    def broadcast_coalesced(tensors, src=0, buffer_size=10 * MB):
        r"""
        Broadcast a sequence of tensors to the default group from rank 0.
        Small tensors are first coalesced into a buffer to reduce the number of
        broadcasts.

        tensors (sequence): tensors to broadcast. Each tensor needs to be on the
                            same GPU.
        src (int): src rank. Default: 0.
        buffer_size (int): maximum size of the buffer for coalescing. Default: 10MB.
        """
        for tensors in _take_tensors(tensors, buffer_size):
            flat_tensors = _flatten_dense_tensors(tensors)
            dist.broadcast(flat_tensors, src)
            for old_t, new_t in zip(tensors, _unflatten_dense_tensors(flat_tensors, tensors)):
                old_t.data = new_t

    def all_reduce_coalesced(tensors, divisor=1, op=ReduceOp.SUM, buffer_size=256 * MB):
        for tensors in _take_tensors(tensors, buffer_size):
            flat_tensors = _flatten_dense_tensors(tensors)
            dist.all_reduce(flat_tensors, op)
            if divisor != 1:
                flat_tensors.div_(divisor)
            for old_t, new_t in zip(tensors, _unflatten_dense_tensors(flat_tensors, tensors)):
                old_t.data = new_t

    # NB: neither nccl nor gloo supports gather, but they are the only options for
    #     gpu distributed, so all gather is all we have

    def all_gather_coalesced(tensors, buffer_size=256 * MB):
        assert dist.get_backend() == dist.dist_backend.NCCL  # gloo gives some weird device error
        world_size = dist.get_world_size()
        rcv_lsts = [[] for _ in range(world_size)]
        for tensors in _take_tensors(tensors, buffer_size):
            flat_tensors = _flatten_dense_tensors(tensors)
            tmp_rcv_lst = [torch.empty_like(flat_tensors) for _ in range(world_size)]
            dist.all_gather(tmp_rcv_lst, flat_tensors)
            for i, rcv_flat_tensors in enumerate(tmp_rcv_lst):
                for rcv_t in _unflatten_dense_tensors(rcv_flat_tensors, tensors):
                    rcv_lsts[i].append(rcv_t)
        return rcv_lsts

    def barrier():
        t = torch.randn((), device='cuda')
        dist.all_reduce(t)
        torch.cuda.synchronize()
