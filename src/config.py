import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    #Hugging face repo
    hf_repo_id:str
    hf_token:str=None

    #Paths
    base_dir:Path=Path(__file__).parent.parent
    data_dir:Path=Path("./data")
    token_dir:Path=Path("./artifacts/tokenized_data")
    model_dir:Path=Path("./artifacts/models")
    cache_dir:Path=Path("./artifacts/cache")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

@lru_cache
def get_settings():
    return Settings()

