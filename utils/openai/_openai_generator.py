import asyncio
from tqdm.asyncio import tqdm_asyncio
from typing import Any, Dict, List, Tuple

from utils.common import load_prompt
from ._get_async_response import get_async_response


class OpenAIGenerator:
    def __init__(self, model_name: str, batch_size: int):
        """Initialization"""
        self.model_name = model_name
        self.batch_size = batch_size
    
    def _select_final_task(self, task: str, task_input: Dict[str, Any]) -> str:
        if task in ['generate_pandas_code', 'generate_vega_lite_code']:
            return task
        
        if (task == 'prompt_with_exec' and task_input['relevant_data']) or \
            (task == 'prompt_with_exec_vis' and task_input['relevant_data'] and not task_input['image_url']):
                return 'prompt_with_exec'
        
        elif (task == 'prompt_with_vis' and task_input['image_url']) or \
            (task == 'prompt_with_exec_vis' and not task_input['relevant_data'] and task_input['image_url']):
                return 'prompt_with_vis'
        
        elif task == 'prompt_with_exec_vis' and task_input['relevant_data'] and task_input['image_url']:
            return 'prompt_with_exec_vis'
        
        return 'prompt_direct'
    
    async def _async_generate(self, task: str, input_set: List[Dict[str, Any]], key_set: List[Any], isImage: bool) -> Tuple[List[Dict[str, Any]], float]:
        """Asynchronous generation"""
        semaphore = asyncio.Semaphore(self.batch_size)

        tasks = [
            get_async_response(
                semaphore=semaphore,
                prompt=(
                    {
                        'text': load_prompt(role='user', task=self._select_final_task(task, task_input)).format(**task_input),
                        'image_url': task_input['image_url'] # task_input['image_url'] is not None
                    }
                    if isImage and task_input['image_url']
                    else {'text': load_prompt(role='user', task=self._select_final_task(task, task_input)).format(**task_input)}
                ),
                model_name=self.model_name,
                key=key
            )
            for task_input, key in zip(input_set, key_set)
        ]

        task_output_set = []

        for _ in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc=f"{f'{self.model_name} - {task}':<30}"):
            task_output = await _
            task_output_set.append(task_output)
        
        cost = sum([output['input_tokens_cost'] + output['output_tokens_cost'] for output in task_output_set])

        return sorted(task_output_set, key=lambda x: x['key']), cost
    
    def generate(self, task: str, input_set: List[Dict[str, Any]], isImage: bool=False) -> Tuple[List[Dict[str, Any]], float]:
        """Generation"""
        task_output_set, cost = asyncio.run(self._async_generate(
            task=task,
            input_set=input_set,
            key_set=list(range(len(input_set))),
            isImage=isImage
        ))

        return task_output_set, cost
