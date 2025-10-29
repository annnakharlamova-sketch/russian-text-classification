import yaml

config = yaml.safe_load(open('configs/experiment_config.yaml', 'r', encoding='utf-8'))
config['data']['corpora']['taiga']['path'] = 'data/taiga_extracted'

with open('configs/experiment_config.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print('Путь для Taiga исправлен на: data/taiga_extracted')