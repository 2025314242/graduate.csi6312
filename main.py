import altair as alt
import argparse
import base64
import csv
import json
import pandas as pd
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

import evaluation
from utils.common import serialize_table
from utils.dataset import load_fetaqa, load_qtsumm
from utils.openai import OpenAIGenerator


def run_task(
    path: str,
    task: str,
    model: OpenAIGenerator,
    input_set: List[Dict[str, Any]],
    isImage: bool=False
    ) -> Tuple[List[Dict[str, Any]], float]:
    try:
        with open(f'{path}/{task}.json', 'r') as file:
            output_set = json.load(file)
        cost = 0
        
    except:
        output_set, cost = model.generate(task=task, input_set=input_set, isImage=isImage)
        with open(f'{path}/{task}.json', 'w') as file:
            json.dump(output_set, file)
        
    return output_set, cost


def _convert_columns_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            continue
    return df

def execute(df: pd.DataFrame, json_str: str) -> str:
    try:
        parsed = json.loads(json_str)
        code = parsed['python_code']
        title = parsed['description']
        
        exec(code)
        rel_df = eval('extract_relevant_data')(df).astype(str)
        relevant_data = serialize_table({
            'title': title,
            'header': rel_df.columns.tolist(),
            'cell': rel_df.values.tolist()
        })
    
    except:
        return None
    
    return relevant_data


def visualize(df: pd.DataFrame, json_str: str, image_url: str) -> Tuple[str, str]:
    try:
        parsed = json.loads(json_str)
        spec = parsed['vega_lite_spec']
        alt_text = parsed['description']
        
        spec['data'] = {'values': df.to_dict(orient='records')}
        chart = alt.Chart.from_dict(spec)
        chart.save(image_url)
    except Exception:
        return None, None
    
    return image_url, alt_text

def _encode_image_to_base64(image_url: str) -> str:
    if not image_url:
        return None
    
    with open(image_url, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    
    return f"data:image/png;base64,{encoded}"


def main(baseline: str, dataset: str):
    if dataset == 'FeTaQA':
        test_set = load_fetaqa()
    elif dataset == 'QTSumm':
        test_set = load_qtsumm()
    else:
        return
    
    model = OpenAIGenerator(
        model_name='gpt-4.1-nano-2025-04-14',
        batch_size=10
    )

    """
    Baseline 1: plain text (Text) - A1
    Baseline 2: execution-based module (Exec) - B1 -> A2
    Ours: execution-based module + visualization-based module (ExecVis) - {B1 + C1} -> A3
    Ablation: visualization-based module (Vis) - C1 -> A4 
    
    A1 : prompt_direct (table + question -> answer)
    A2 : prompt_with_exec (table + question + execution -> answer)
    A3 : prompt_with_exec_vis (table + question + execution + visualization -> answer)
    A4 : prompt_with_vis (table + question + visualization -> answer)
    B1 : generate_pandas_code (table + question -> pandas code -> execution)
    C1 : generate_vega_lite_code (table + question -> vega-lite code -> visualization)
    """
    
    path = f'buffer/{dataset.lower()}'
    
    if baseline == 'Text':
        # A1 : prompt_direct (table + question -> answer)
        task = 'prompt_direct'
        input_set = [
            {
                'question': data['question'],
                'serialized_table': serialize_table(data['table'])
            }
            for data in test_set
        ]
        output_set, cost = run_task(path=path, task=task, model=model, input_set=input_set)
        print(f"Cost - {task}: ${cost}.")
        
        final_output_set = output_set
    
    elif baseline == 'Exec':
        # B1 : generate_pandas_code (table + question -> pandas code -> execution)
        task = 'generate_pandas_code'
        input_set = [
            {
                'question': data['question'],
                'serialized_table': serialize_table(data['table'])
            }
            for data in test_set
        ]
        
        output_set, cost = run_task(path=path, task=task, model=model, input_set=input_set)
        print(f"Cost - {task}: ${cost}.")
        
        # EXECUTION #
        execution_result_set = [
            execute(
                df=_convert_columns_to_numeric(pd.DataFrame(data['table']['cell'], columns=data['table']['header'])),
                json_str=res['response']
            )
            for data, res in tqdm(zip(test_set, output_set), total=len(test_set))
        ]

        # A2 : prompt_with_exec (table + question + execution -> answer)
        task = 'prompt_with_exec'
        input_set = [
            {
                'question': data['question'],
                'serialized_table': serialize_table(data['table']),
                'relevant_data': exec_res
            }
            for data, exec_res in zip(test_set, execution_result_set)
        ]
        output_set, cost = run_task(path=path, task=task, model=model, input_set=input_set)
        print(f"Cost - {task}: ${cost}.")
        
        final_output_set = output_set
    
    elif baseline == 'ExecVis':
        # B1 : generate_pandas_code (table + question -> pandas code -> execution)
        task = 'generate_pandas_code'
        input_set = [
            {
                'question': data['question'],
                'serialized_table': serialize_table(data['table'])
            }
            for data in test_set
        ]
        output_set, cost = run_task(path=path, task=task, model=model, input_set=input_set)
        print(f"Cost - {task}: ${cost}.")
        
        # EXECUTION #
        execution_result_set = [
            execute(
                df=_convert_columns_to_numeric(pd.DataFrame(data['table']['cell'], columns=data['table']['header'])),
                json_str=res['response']
            )
            for data, res in tqdm(zip(test_set, output_set), total=len(test_set))
        ]
        
        # C1 : generate_vega_lite_code (table + question -> vega-lite code -> visualization)
        task = 'generate_vega_lite_code'
        input_set = [
            {
                'question': data['question'],
                'serialized_table': serialize_table(data['table'])
            }
            for data in test_set
        ]
        output_set, cost = run_task(path=path, task=task, model=model, input_set=input_set)
        print(f"Cost - {task}: ${cost}.")
        
        # VISUALIZATION #
        visualization_result_set = [
            visualize(
                df=_convert_columns_to_numeric(pd.DataFrame(data['table']['cell'], columns=data['table']['header'])),
                json_str=res['response'],
                image_url=f'images/{dataset.lower()}_{idx:04d}.png'
            )
            for idx, (data, res) in tqdm(enumerate(zip(test_set, output_set)), total=len(test_set))
        ]
    
        # A3 : prompt_with_exec_vis (table + question + execution + visualization -> answer)
        task = 'prompt_with_exec_vis'
        input_set = [
            {
                'question': data['question'],
                'serialized_table': serialize_table(data['table']),
                'relevant_data': exec_res,
                'alt_text': alt_text,
                'image_url': _encode_image_to_base64(image_url),
            }
            for data, exec_res, (image_url, alt_text) in zip(test_set, execution_result_set, visualization_result_set)
        ]
        output_set, cost = run_task(path=path, task=task, model=model, input_set=input_set, isImage=True)
        print(f"Cost - {task}: ${cost}.")
        
        final_output_set = output_set
    
    elif baseline == 'Vis':
        # C1 : generate_vega_lite_code (table + question -> vega-lite code -> visualization)
        task = 'generate_vega_lite_code'
        input_set = [
            {
                'question': data['question'],
                'serialized_table': serialize_table(data['table'])
            }
            for data in test_set
        ]
        output_set, cost = run_task(path=path, task=task, model=model, input_set=input_set)
        print(f"Cost - {task}: ${cost}.")
        
        # VISUALIZATION #
        visualization_result_set = [
            visualize(
                df=_convert_columns_to_numeric(pd.DataFrame(data['table']['cell'], columns=data['table']['header'])),
                json_str=res['response'],
                image_url=f'images/{dataset.lower()}_{idx:04d}.png'
            )
            for idx, (data, res) in tqdm(enumerate(zip(test_set, output_set)), total=len(test_set))
        ]
        
        # A4 : prompt_with_vis (table + question + visualization -> answer)
        task = 'prompt_with_vis'
        input_set = [
            {
                'question': data['question'],
                'serialized_table': serialize_table(data['table']),
                'alt_text': alt_text,
                'image_url': _encode_image_to_base64(image_url),
            }
            for data, (image_url, alt_text) in zip(test_set, visualization_result_set)
        ]
        output_set, cost = run_task(path=path, task=task, model=model, input_set=input_set, isImage=True)
        print(f"Cost - {task}: ${cost}.")
        
        final_output_set = output_set
    
    else:
        return
    
    device = 'cpu'
    
    table_set = [data['table'] for data in test_set]
    target = [data['answer'] for data in test_set]
    output = [res['response'] for res in final_output_set]
    
    sacrebleu_set = evaluation.SacreBLEU(target=target, output=output)
    print("[Done] SacreBLEU Evaluation.")

    rouge_l_set = evaluation.ROUGE_L(target=target, output=output)
    print("[Done] ROUGE-L Evaluation.")

    meteor_set = evaluation.METEOR(target=target, output=output)
    print("[Done] METEOR Evaluation.")

    bertscore_set = evaluation.BERTScore(target=target, output=output, device=device)
    print("[Done] BERTScore Evaluation.")

    a3cu_set = evaluation.A3CU(target=target, output=output, device=device)
    print("[Done] A3CU Evaluation.")
    
    tapas_acc_set = evaluation.TAPAS_Acc(table_set=table_set, output=output, device=device)
    print("[Done] TAPAS-Acc Evaluation.")
    
    results = []
    
    for sacrebleu, rouge_l, meteor, bertscore, a3cu, tapas_acc, res in zip(
            sacrebleu_set, rouge_l_set, meteor_set, bertscore_set, a3cu_set, tapas_acc_set, output
        ):
        results.append({
            'sacrebleu': sacrebleu if res != '' else 0.0,
            'rouge_l': rouge_l * 100 if res != '' else 0.0,
            'meteor': meteor * 100 if res != '' else 0.0,
            'bertscore': bertscore * 100 if res != '' else 0.0,
            'a3cu': a3cu * 100 if res != '' else 0.0,
            'tapas_acc': tapas_acc * 100 if res != '' else 0.0
        })
    
    print("[Done] Evaluation.")
    
    with open(f'results/{dataset.lower()}_{baseline.lower()}.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['sacrebleu', 'rouge_l', 'meteor', 'bertscore', 'a3cu', 'tapas_acc'])
        writer.writeheader()
        writer.writerows(results)

    ###


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', choices=['Text', 'Exec', 'ExecVis', 'Vis'])
    parser.add_argument('--dataset', choices=['FeTaQA', 'QTSumm'])
    args, _ = parser.parse_known_args()
    
    main(
        baseline=args.baseline,
        dataset=args.dataset
    )
