from evaluate import load
from tqdm import tqdm
from typing import List


def BERTScore(
    target: List[str],
    output: List[str],
    device: str
    ) -> List[float]:
    """bertscore

    [Params]
    target : List[str]
    output : List[str]
    device : str
    
    [Return]
    score_list : List[float]
    """
    bertscore = load('bertscore')

    batch_size = 32
    total_batches = (len(output) + batch_size - 1) // batch_size

    f1_score_list = []
    for i in tqdm(range(total_batches)):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(output))

        results = bertscore.compute(
            predictions=output[start:end],
            references=target[start:end],
            lang='en',
            device=device
        )
        f1_score_list.extend(results['f1'])

    return f1_score_list
