from dataclasses import dataclass

@dataclass
class TrainConfig:
    random_seed: int = 1009
    template: str = 'plain'# [plain, title, detail]
    vlmModel: str = 'qwen'# [qwen, blip, gema, lava]
    data: str = 'baby'# ["baby", "sport", "cloth"]
    