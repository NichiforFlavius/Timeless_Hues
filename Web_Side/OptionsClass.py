from dataclasses import dataclass, field
from datetime import datetime


def get_unique_image_name():
    return f'revived_image_{datetime.now().strftime("%d_%m_%Y_%H_%M")}.png'


@dataclass
class ImageOptions:
    contrast: float = field(default=0.1, metadata={'description': 'The value for the applied contrast'})
    exposure: float = field(default=0.1, metadata={'description': 'The value for the applied exposure'})
    color: float = field(default=0.1, metadata={'description': 'The value for the applied color'})
    sharpness: float = field(default=0.1, metadata={'description': 'The value for the applied sharpness'})

    horizontal_flip: bool = field(default=False, metadata={'description': 'The flag that states if the image is flipped horizontally'})
    vertical_flip: bool = field(default=False, metadata={'description': 'The flag that states if the image is flipped vertically'})
    blur: float = field(default=0.0, metadata={'description': 'The value for the applied blur'})

    download_name: str = field(default_factory=get_unique_image_name)

    custom_messages: dict = field(default_factory=lambda: {
        "contrast": f"The contrast can only have values between: 0.0-10.0",
        "exposure": "The exposure can only have values between: 0.0-10.0",
        "color": "The color can only have values between: 0.0-10.0",
        "sharpness": "The sharpness can only have values between: 0.0-10.0",
        "horizontal_flip": "The horizontal flip must be a boolean value",
        "vertical_flip": "The vertical flip must be a boolean value",
        "blur": "The blur can only have values between 0.0-10.0",
    })

    def __post_init__(self):
        self.validate_contrast()
        self.validate_exposure()
        self.validate_color()
        self.validate_sharpness()
        self.validate_h_flip()
        self.validate_v_flip()
        self.validate_blur()

    def validate_contrast(self):
        if not 0.1 <= self.contrast <= 10.0:
            raise ValueError(self.custom_messages["contrast"])

    def validate_exposure(self):
        if not 0.1 <= self.exposure <= 10.0:
            raise ValueError(self.custom_messages["exposure"])

    def validate_color(self):
        if not 0.1 <= self.color <= 10.0:
            raise ValueError(self.custom_messages["color"])

    def validate_sharpness(self):
        if not 0.1 <= self.sharpness <= 10.0:
            raise ValueError(self.custom_messages["sharpness"])

    def validate_h_flip(self):
        if not isinstance(self.horizontal_flip, bool):
            raise TypeError(self.custom_messages["horizontal_flip"])

    def validate_v_flip(self):
        if not isinstance(self.vertical_flip, bool):
            raise TypeError(self.custom_messages["vertical_flip"])

    def validate_blur(self):
        if not 0.0 <= self.blur <= 10.0:
            raise ValueError(self.custom_messages["blur"])
