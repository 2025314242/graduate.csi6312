import yaml


if __name__ == 'utils.openai._load_config':
    OPENAI_API_KEY =  yaml.load(open('.config.yaml'), Loader=yaml.FullLoader)['OPENAI_API_KEY']

    PRICING = {
        'gpt-4.1-nano-2025-04-14': {'input_token_price': 0.100/1e6, 'output_token_price': 0.400/1e6},
    }
