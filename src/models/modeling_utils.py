from pathlib import Path

from src.checkpoint import auto_split_huggingface_checkpoints, auto_split_consolidate_checkpoints
from src.checkpoint_baichuan import auto_split_huggingface_checkpoints_for_baichuan
from src.utils import set_barrier


def auto_split_or_merge_checkpoints(
        ckpt_dir: str,
        model_parallel_world_size: int,
        global_rank: int
) -> str:
    checkpoints = sorted(Path(ckpt_dir).glob("consolidated.*.pth"))
    if len(checkpoints) == 0:  # splitting
        ckpt_dir = auto_split_huggingface_checkpoints(
            ckpt_dir,
            model_parallel_world_size=model_parallel_world_size,
            global_rank=global_rank
        )
    elif len(checkpoints) != model_parallel_world_size:
        ckpt_dir = auto_split_consolidate_checkpoints(
            ckpt_dir,
            model_parallel_world_size=model_parallel_world_size,
            global_rank=global_rank
        )
    set_barrier()
    return ckpt_dir


def auto_split_or_merge_checkpoints_for_baichuan(
        ckpt_dir: str,
        model_parallel_world_size: int,
        global_rank: int
) -> str:
    checkpoints = sorted(Path(ckpt_dir).glob("consolidated.*.pth"))
    if len(checkpoints) == 0:  # splitting
        ckpt_dir = auto_split_huggingface_checkpoints_for_baichuan(
            ckpt_dir,
            model_parallel_world_size=model_parallel_world_size,
            global_rank=global_rank
        )
    set_barrier()
    return ckpt_dir
