from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from tqdm import tqdm
from typing import List


def METEOR(
    target: List[str],
    output: List[str]
    ) -> List[float]:
    """meteor

    [Params]
    target : List[str]
    output : List[str]
    
    [Return]
    score_list : List[float]
    """
    score_list = []
    for ref, hyp in tqdm(zip(target, output)):
        score_list.append(meteor_score(references=[word_tokenize(ref.lower())], hypothesis=word_tokenize(hyp.lower())))
    
    return score_list
