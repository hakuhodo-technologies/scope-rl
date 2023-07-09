import torch


def format_runtime(start: int, finish: int):
    runtime = finish - start
    hour = int(runtime // 3600)
    min = int((runtime // 60) % 60)
    sec = int(runtime % 60)
    return f"{hour}h.{min}m.{sec}s"


def torch_seed(random_state: int, device=str):
    if device == "cuda:0":
        torch.cuda.manual_seed(random_state)

    torch.manual_seed(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
