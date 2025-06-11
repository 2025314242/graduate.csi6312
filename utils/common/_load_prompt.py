from typing import Literal


def load_prompt(
    role: Literal['assistant', 'system', 'user'],
    task: str
    ) -> str:
    """Load prompt

    [Params]
    role : Literal['assistant', 'system', 'user']
    task : str

    [Return]
    prompt : str
    """
    prompt = open(f'prompts/{role}/{task}.txt', 'r').read()

    return prompt
