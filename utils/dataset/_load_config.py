import yaml


if __name__ == 'utils.dataset._load_config':
    HUGGINGFACE_ACCESS_TOKEN =  yaml.load(open('.config.yaml'), Loader=yaml.FullLoader)['HUGGINGFACE_ACCESS_TOKEN']
