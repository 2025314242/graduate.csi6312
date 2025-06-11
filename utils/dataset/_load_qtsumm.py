from datasets import load_dataset
from huggingface_hub import login
from typing import Literal, Union

from ._load_config import HUGGINGFACE_ACCESS_TOKEN


class QTSummTestSet:
    def __init__(self):
        """Initialization"""
        self._raw_test_set = load_dataset('yale-nlp/QTSumm')['test']
        self._test_set = self._load_test_set()
    
    def _load_test_set(self):
        test_set = []
        
        for data in self._raw_test_set:
            table = {
                'title': data['table']['title'],
                'header': data['table']['header'],
                'cell': data['table']['rows']
            }
            
            instance = {
                'table': table,
                'question': data['query'],
                'answer': data['summary']
            }
            
            test_set.append(instance)
        
        return test_set
    
    def __len__(self) -> int:
        """Test set size"""
        return len(self._test_set)
    
    def __getitem__(self, key: Union[int, slice]):
        """Test set data"""
        return self._test_set[key]
    
    def __str__(self) -> Literal['FeTaQA test set']:
        """Representation"""
        return 'FeTaQA test set'


def load_qtsumm() -> QTSummTestSet:
    """Load QTSumm test set

    [Return]
    fetaqa : QTSummTestSet
    """
    global qtsumm
    if qtsumm is None:
        qtsumm = QTSummTestSet()
    
    return qtsumm


if __name__ == 'utils.dataset._load_qtsumm':
    login(HUGGINGFACE_ACCESS_TOKEN)
    
    qtsumm = None
