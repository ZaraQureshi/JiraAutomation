import os
from pathlib import Path
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings
ROOT_DIR = Path(__file__).resolve().parent.parent
class Settings(BaseSettings):
    # use optional to make these fields optional
    hf_repo_id: Optional[str] = None
    hf_token: Optional[str] = None
    api_url: Optional[str] = None
    dataset_repo_id:Optional[str]=None

    #Paths
    base_dir:Path=ROOT_DIR
    data_dir:Path=Path("./data")
    token_dir:Path=Path("./artifacts/tokenized_data")
    model_dir:Path=Path("./artifacts/models")
    cache_dir:Path=Path("./artifacts/cache")

    class Config:
        env_file = str(ROOT_DIR/".env")
        env_file_encoding = "utf-8"
        case_sensitive = False

@lru_cache
def get_settings():
    return Settings()

