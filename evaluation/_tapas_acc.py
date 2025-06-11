import pandas as pd
import torch
import warnings
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import TapasForSequenceClassification, TapasTokenizer
from typing import Any, Dict, List


class EvalData(Dataset):
    def __init__(self, tokenizer: Any, table_set: List[Dict[str, Any]], data: List[str]):
        """Initialization"""
        self.Data = self.load_data(table_set, data)
        self.len = len(self.Data)
        self.tokenizer = tokenizer
        self.fail = 0
    
    def load_data(self, table_set: List[Dict[str, Any]], data: List[str]) -> List[Dict[str, Any]]:
        """Loading"""
        loaded_data = []
        for tab, sample in zip(table_set, data):
            loaded_data.append({'pred_table': tab, 'pred_insight': sample})
        
        return loaded_data

    def read_data(self, data: Dict[str, Any]):
        """Reading"""
        pred = data['pred_insight']
        table = pd.DataFrame(data['pred_table']['cell'], columns=data['pred_table']['header']).fillna('').astype(str)

        return table, pred
    
    def encode(self, table: pd.DataFrame, pred: str) -> torch.Tensor:
        """Encoding"""
        try:
            return self.tokenizer(
                table=table,
                queries=pred,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
        except:
            self.fail += 1
            return self.tokenizer(
                table=table,
                queries='', # too long insight
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )

    def __getitem__(self, idx):
        """Item"""
        concat_table, pred = self.read_data(self.Data[idx])
        return self.encode(concat_table, pred)

    def __len__(self):
        """Length"""
        return self.len


def TAPAS_Acc(
        table_set: List[Dict[str, Any]],
        output: List[str],
        device: str
    ) -> List[float]:
    """tapas-acc

    [Params]
    table_set  : List[Dict[str, Any]]
    output     : List[str]
    device     : str

    [Return]
    score_list : List[float]
    """
    model_name = 'google/tapas-base-finetuned-tabfact'
    batch_size = 4
    device = device

    # Ignore warning message
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    tokenizer = TapasTokenizer.from_pretrained(model_name)
    model = TapasForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    eval_data = EvalData(tokenizer, table_set=table_set, data=output)
    eval_dataloader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=1)

    score_list = []

    for batch in tqdm(eval_dataloader):
        input_ids = batch['input_ids'].squeeze(1).to(device)
        attention_mask = batch['attention_mask'].squeeze(1).to(device)
        token_type_ids = batch['token_type_ids'].squeeze(1).to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        model_preds = outputs.logits.argmax(-1)

        instance_scores = model_preds.float().tolist()
        score_list.extend(instance_scores)

    return score_list
