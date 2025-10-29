import os

print('=== СТРУКТУРА TAIGA_EXTRACTED ===')
taiga_path = 'data/taiga_extracted'

for root, dirs, files in os.walk(taiga_path):
    print(f'Папка: {root}')
    if files:
        for file in files[:5]:  # покажем первые 5 файлов
            print(f'  Файл: {file}')
    else:
        print('  (пусто)')