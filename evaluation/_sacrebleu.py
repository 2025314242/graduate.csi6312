import sacrebleu
from tqdm import tqdm
from typing import List


def SacreBLEU(
    target: List[str],
    output: List[str]
    ) -> List[float]:
    """sacrebleu

    [Params]
    target : List[str]
    output : List[str]
    
    [Return]
    score_list : List[float]
    """
    references = [[ref] for ref in target]
    references = list(zip(*references))

    batch_size = 1
    total_batches = (len(output) + batch_size - 1) // batch_size
    
    score_list = []
    for i in tqdm(range(total_batches)):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(output))

        score = sacrebleu.corpus_bleu(
            hypotheses=output[start:end],
            references=[ref[start:end] for ref in references],
            lowercase=True
        ).score
        score_list.append(score)
    
    return score_list
