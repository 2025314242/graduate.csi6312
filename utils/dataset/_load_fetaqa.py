from datasets import load_dataset
from typing import Literal, Union


class FeTaQATestSet:
    def __init__(self):
        """Initialization"""
        self._raw_test_set = load_dataset('DongfuJiang/FeTaQA')['test']
        self._test_set = self._load_test_set()
    
    def _load_test_set(self):
        test_set = []
        
        for data in self._raw_test_set:
            table = {
                'title': f"{data['table_page_title']} | {data['table_section_title']}",
                'header': data['table_array'][0],
                'cell': data['table_array'][1:]
            }
            
            instance = {
                'table': table,
                'question': data['question'],
                'answer': data['answer']
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


def load_fetaqa() -> FeTaQATestSet:
    """Load FeTaQA test set
    [Return]
    fetaqa : FeTaQATestSet
    """
    global fetaqa
    if fetaqa is None:
        fetaqa = FeTaQATestSet()
    
    return fetaqa


if __name__ == 'utils.dataset._load_fetaqa':
    fetaqa = None
