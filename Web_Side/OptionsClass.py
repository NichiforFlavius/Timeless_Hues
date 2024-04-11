from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from rich.console import Console
from tqdm import tqdm


def get_unique_image_name():
    return f'revived_image_{datetime.now().strftime("%d_%m_%Y_%H_%M")}.png'


@dataclass
class ImageOptions:
    contrast: float = field(default=0, metadata={'description': 'The value for the applied contrast'})
    exposure: float = field(default=0, metadata={'description': 'The value for the applied exposure'})
    color: float = field(default=0, metadata={'description': 'The value for the applied color'})
    sharpness: float = field(default=0, metadata={'description': 'The value for the applied sharpness'})

    horizontal_flip: bool = field(default=False, metadata={'description': 'The flag that states if the image is flipped horizontally'})
    vertical_flip: bool = field(default=False, metadata={'description': 'The flag that states if the image is flipped vertically'})
    grayscale: bool = field(default=False, metadata={'description': 'The flag that states if the image is in grayscale (CIELAB space)'})
    blur: float = field(default=0, metadata={'description': 'The value for the applied blur'})

    download_name: str = field(default_factory=get_unique_image_name)

    custom_messages: dict = field(default_factory=lambda: {
        "contrast": f"Default error message",
        "exposure": "Default error message",
        "color": "Default error message",
        "sharpness": "Default error message",
        "horizontal_flip": "Default error message",
        "vertical_flip": "Default error message",
        "grayscale": "Default error message",
        "blur": "Default error message",
    })

    def __post_init__(self):
        self.validate_contrast()
        self.validate_exposure()
        self.validate_color()
        self.validate_sharpness()
        self.validate_h_flip()
        self.validate_v_flip()
        self.validate_grayscale()
        self.validate_blur()

    def validate_contrast(self):
        ...

    def validate_exposure(self):
        ...

    def validate_color(self):
        ...

    def validate_sharpness(self):
        ...

    def validate_h_flip(self):
        ...

    def validate_v_flip(self):
        ...

    def validate_grayscale(self):
        ...

    def validate_blur(self):
        ...
