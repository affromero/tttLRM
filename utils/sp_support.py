import torch
import torch.distributed as dist
from torch.autograd import Function

_SP_GROUP = None

class _AllReduce(Function):
    @staticmethod
    def forward(ctx, op, group, tensor):
        ctx.group = group
        ctx.op = op
        tensor = tensor.clone().contiguous()
        dist.all_reduce(tensor, op=op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None) + (_AllReduce.apply(ctx.op, ctx.group, grad_output),)

@torch._dynamo.allow_in_graph
def sp_all_reduce(tensor, op=dist.ReduceOp.SUM):
    if _SP_GROUP is None:
        return tensor
    return _AllReduce.apply(op, _SP_GROUP, tensor)

def init_sp_group(sp_size):
    global _SP_GROUP

    if not dist.is_initialized():
        _SP_GROUP = None
        return

    assert dist.get_world_size() % sp_size == 0, f"world size {dist.get_world_size()} is not divisible by sp size {sp_size}"

    sp_group, sp_groups = dist.new_subgroups(group_size=sp_size)
    _SP_GROUP = sp_group

def is_sp():
    return _SP_GROUP is not None

def get_sp_rank():
    if _SP_GROUP is None:
        return 0
    return dist.get_rank(_SP_GROUP)

def get_sp_world_size():
    if _SP_GROUP is None:
        return 1
    return dist.get_world_size(_SP_GROUP)

def get_sp_replicas():
    if _SP_GROUP is None:
        return 1
    return dist.get_world_size() // get_sp_world_size()

def get_sp_replica_id():
    if _SP_GROUP is None:
        return 0
    return dist.get_rank() // get_sp_world_size()

def sp_broadcast_different_size(x: torch.Tensor):
    if _SP_GROUP is None:
        return x, tuple(x.shape)

    # Get the source global rank for the broadcast
    src_rank = dist.get_global_rank(_SP_GROUP, 0)

    # Step 1: Broadcast the shape
    if dist.get_rank() == src_rank:
        shape_tensor = torch.tensor(x.shape, dtype=torch.int64, device=x.device)
    else:
        shape_tensor = torch.empty(x.dim(), dtype=torch.int64, device=x.device)

    dist.broadcast(shape_tensor, src=src_rank, group=_SP_GROUP)
    new_shape = tuple(shape_tensor.tolist())

    # Step 2: Resize x to match the broadcasted shape if not on source rank
    if dist.get_rank() != src_rank:
        x = x.new_empty(new_shape)

    # Step 3: Broadcast the tensor data
    dist.broadcast(x, src=src_rank, group=_SP_GROUP)
    return x, new_shape

def sp_broadcast(x: torch.Tensor):
    if _SP_GROUP is None:
        return x

    if dist.get_rank(_SP_GROUP) != 0:
        pass

    # Note: in torch 2.7, there is group_src that you can just specify the group's rank.
    #       we are still at torch 2.5, thus need to use the global rank.
    dist.broadcast(x, src=dist.get_global_rank(_SP_GROUP, 0), group=_SP_GROUP)

    return x

def slice_tensor(x, dim, start, end):
    slices = [slice(None)] * x.dim()  # create slices for all dims
    slices[dim] = slice(start, end)   # set slice for the desired dim
    return x[tuple(slices)]

def pad0_tensor(x, dim, pad_len):
    if pad_len <= 0:
        return x  # no padding needed

    # Create a zero tensor of the same dtype and device as x
    pad_shape = list(x.shape)
    pad_shape[dim] = pad_len
    pad_tensor = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)

    # Concatenate along the specified dimension
    return torch.cat([x, pad_tensor], dim=dim)

def gather_scatter(x, gather_dim=2, scatter_dim=1, process_group=None):
    if process_group is None:
        return x

    from torch.distributed.nn.functional import all_gather
    # [b, N, s, d]
    all_x = all_gather(x, group=process_group)
    local_rank = process_group.rank()

    scatter_size = process_group.size()
    scatter_len = x.size(scatter_dim)
    assert scatter_len % scatter_size == 0, f"scatter_len {scatter_len} is not divisible by scatter_size {scatter_size}"
    scatter_stride = scatter_len // scatter_size

    # [b, n, s, d]
    all_x = [
        slice_tensor(x, scatter_dim, local_rank * scatter_stride, (local_rank + 1) * scatter_stride)
        for x in all_x
    ]

    # [b, n, S, d]
    all_x = torch.cat(all_x, dim=gather_dim)

    return all_x


def sp_gather_scatter(x, gather_dim=2, scatter_dim=1):
    if _SP_GROUP is None:
        return x
    return gather_scatter(x, gather_dim=gather_dim, scatter_dim=scatter_dim, process_group=_SP_GROUP)


def local_scatter(x, scatter_dim=1, process_group=None):
    if process_group is None:
        return x

    local_rank = process_group.rank()

    scatter_size = process_group.size()
    scatter_len = x.size(scatter_dim)
    assert scatter_len % scatter_size == 0, f"scatter_len {scatter_len} is not divisible by scatter_size {scatter_size}"
    scatter_stride = scatter_len // scatter_size

    # [b, n, s, d]
    all_x = slice_tensor(x, scatter_dim, local_rank * scatter_stride, (local_rank + 1) * scatter_stride)

    return all_x


def sp_local_scatter(x, scatter_dim=1):
    if _SP_GROUP is None:
        return x
    return local_scatter(x, scatter_dim=scatter_dim, process_group=_SP_GROUP)


def sp_input_broadcast_scatter(x, scatter_dim=1, different_size=False):
    """
    Note:
        1. need all rank have the same size and type of x.
        2. it will auto-pad and shard among the scatter_dim given the sp world size.
    """
    if _SP_GROUP is None:
        if different_size:
            return x, tuple(x.shape)
        return x

    if different_size:
        # [b, n, s, d]
        x, new_shape = sp_broadcast_different_size(x)
    else:
        x = sp_broadcast(x)

    # [b, n, S, d]
    sp_world_size = get_sp_world_size()

    # If the input can not be divided by the sp size, pad it
    if x.size(scatter_dim) % sp_world_size != 0:
        pad_len = sp_world_size - (x.size(scatter_dim) % sp_world_size)
        x = pad0_tensor(x, scatter_dim, pad_len)

    # [b, n, s, d]
    x = sp_local_scatter(x, scatter_dim=scatter_dim)

    if different_size:
        return x, new_shape
    else:
        return x

def sp_all_gather(x, gather_dim=2, length=-1):
    """
    Note:
        1. need all rank have the same size and type of x.
    """
    if _SP_GROUP is None:
        if length != -1:
            x = slice_tensor(x, gather_dim, 0, length)
        return x

    from torch.distributed.nn.functional import all_gather

    # [b, n, S, d]
    x = all_gather(x, group=_SP_GROUP)      # list of x's
    x = torch.cat(x, dim=gather_dim)

    if length != -1:
        assert x.size(gather_dim) >= length, f"gathered tensor size {x.size(gather_dim)} is less than the given length {length}"
        x = slice_tensor(x, gather_dim, 0, length)

    return x
