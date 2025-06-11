from langchain_openai import ChatOpenAI

from ._load_config import OPENAI_API_KEY


def load_llm(model_name: str) -> ChatOpenAI:
    """Load OpenAI LLM

    [Param]
    model_name : str

    [Return]
    llm : ChatOpenAI
    """
    if model_name not in OPENAI_LLM_BUFFER:
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            api_key=OPENAI_API_KEY
        )
        OPENAI_LLM_BUFFER[model_name] = llm
    
    else:
        llm = OPENAI_LLM_BUFFER[model_name]
    
    return llm


if __name__ == 'utils.openai._load_llm':
    OPENAI_LLM_BUFFER = dict()
