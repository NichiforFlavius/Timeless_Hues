import yaml

from pathlib import Path
from rich.console import Console

CONSOLE = Console(color_system='truecolor')


def __load_model_config(config_file_path: str | Path) -> dict[str, any] | None:
    """
        Loads a configuration file in yaml format and returns its contents, if successful.

        Args:
            config_file_path (str | Path): path to config file in yaml format using a string or pathlib.Path object

        Returns:
            dict(str, any) as the content of the config file, or None if any errors occurred when loading the file.
    """

    if isinstance(config_file_path, str):
        config_file_path = Path(config_file_path)

    try:
        config_file_content = yaml.safe_load(config_file_path.open(mode='r'))
        return config_file_content
    except yaml.YAMLError:
        CONSOLE.print(f'Error loading [magenta]{config_file_path}[/magenta]\n')
    return None


def load_onnx_model_info(config_file_path: str | Path, model_type: str = "large"):
    """

    Args:
        config_file_path:
        model_type:
    Returns:

    """

    config_data = __load_model_config(config_file_path)
    model_path = config_data[model_type]['model_path']
    model_size = config_data[model_type]['model_size']
    return model_path, model_size
