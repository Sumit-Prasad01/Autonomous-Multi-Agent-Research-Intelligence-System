from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir : Path
    dataset_name : Path
    unzip_dir : Path


@dataclass
class DataTransformationConfig:
    root_dir : Path
    data_path : Path
    tokenizer_name : Path