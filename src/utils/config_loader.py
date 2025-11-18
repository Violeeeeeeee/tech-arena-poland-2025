import yaml
import argparse
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads configuration from a YAML file.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        exit(1)

def parse_main_args() -> argparse.Namespace:
    """
    Parses command-line arguments, primarily to specify the config file path.
    """
    parser = argparse.ArgumentParser(description="Run Sentence Embedding Distillation Experiment.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base_config.yaml',
        help='Path to the YAML configuration file.'
    )
    return parser.parse_args()

if __name__ == '__main__':
    # Тест: завантаження та друк
    test_config_path = '../../configs/test_config.yaml' # Припустімо, що файл тут
    print(f"Testing config loading from {test_config_path}...")
    try:
        config = load_config(test_config_path)
        print("Successfully loaded config:")
        print(config)
    except Exception as e:
        print(f"Test failed: {e}")
