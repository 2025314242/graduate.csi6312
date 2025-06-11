import asyncio
import time
import traceback
from langchain.schema import HumanMessage
from typing import Any, Dict, Literal

from ._load_config import PRICING
from ._load_llm import load_llm


async def get_async_response(
    semaphore: asyncio.Semaphore,
    prompt: Dict[Literal['text', 'image_url'], str],
    model_name: str,
    key: Any
    ) -> Dict[str, Any]:
    """Get asynchronous OpenAI response

    [Params]
    semaphore     : asyncio.Semaphore
    prompt   : Dict[Literal['text', 'image_url'], str]
    model_name    : str
    key           : Any

    [Return]
    instance : Dict[str, Any]
    e.g. dict_keys(['prompt', 'response', 'input_tokens_cost', 'output_tokens_cost', 'key'])
    """
    llm = load_llm(model_name=model_name)

    input_token_price = PRICING[model_name]['input_token_price']
    output_token_price = PRICING[model_name]['output_token_price']

    async with semaphore:
        while True:
            try:
                message_content = [
                    {'type': 'text', 'text': prompt['text']},
                    {'type': 'image_url', 'image_url': {'url': prompt['image_url']}}
                ] if prompt.get("image_url") else prompt['text']
                human_message = HumanMessage(message_content)

                response = await llm.ainvoke([human_message])

                input_tokens = response.usage_metadata['input_tokens']
                output_tokens = response.usage_metadata['output_tokens']

                input_tokens_cost = input_tokens * input_token_price
                output_tokens_cost = output_tokens * output_token_price
                
                break
            
            except Exception:
                print(f"[{key}] {traceback.format_exc()}")
                
                time.sleep(10)

    return {
        'prompt': prompt,
        'response': response.content.replace('\n', ' ').strip(),
        'input_tokens_cost': input_tokens_cost,
        'output_tokens_cost': output_tokens_cost,
        'key': key
    }
