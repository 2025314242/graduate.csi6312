from nltk.tokenize import word_tokenize
from tqdm import tqdm
from typing import List


def lcs(
        tokens1: List[str],
        tokens2: List[str]
    ) -> int:
    """LCS (Longest Common Sequence)"""
    m, n = len(tokens1), len(tokens2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens1[i - 1] == tokens2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def ROUGE_L(
    target: List[str],
    output: List[str]
    ) -> List[float]:
    """rouge-l

    [Params]
    target : List[str]
    output : List[str]
    
    [Return]
    score_list : List[float]
    """
    f1_score_list = []
    for ref, hyp in tqdm(zip(target, output)):
        ref_tokens = word_tokenize(ref.lower())
        hyp_tokens = word_tokenize(hyp.lower())

        lcs_len = lcs(ref_tokens, hyp_tokens)
        ref_len = len(ref_tokens)
        hyp_len = len(hyp_tokens)

        precision = lcs_len / hyp_len if hyp_len > 0 else 0.0
        recall = lcs_len / ref_len if ref_len > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        f1_score_list.append(f1_score)
    
    return f1_score_list
