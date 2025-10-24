from dataclasses import dataclass

@dataclass
class TrainConfig:
    random_seed: int = 1009
    template: str = 'title'# [plain, title, detail]
    vlmModel: str = 'qwen'# [qwen, lama, gema, lava]
    data: str = 'baby'# ["baby", "sport", "cloth"]
    