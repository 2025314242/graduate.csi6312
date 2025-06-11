import autoacu
from typing import List


def A3CU(
    target: List[str],
    output: List[str],
    device: str
    ) -> List[float]:
    """a3cu

    [Params]
    target : List[str]
    output : List[str]
    device : str
    
    [Return]
    score_list : List[float]
    """
    if device == 'cpu':
        a3cu = autoacu.A3CU(cpu=True)
    else:
        a3cu = autoacu.A3CU(device=int(device.split(':')[1]))
    _, _, f1_score_list = a3cu.score(
        references=target,
        candidates=output,
        batch_size=16
    )

    return f1_score_list
