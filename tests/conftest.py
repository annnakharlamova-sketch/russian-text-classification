"""
Конфигурация pytest
"""

import sys
import os

# Добавляем корень проекта в Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)