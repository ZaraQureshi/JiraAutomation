from huggingface_hub import HfApi, upload_folder, snapshot_download, hf_hub_download
from pathlib import Path
from typing import Optional

class HFStorage:
    def __init__(self):
        self.repo_id=process.env.HF_REPO_ID
        self.token=process.env.HF_TOKEN
        self.api=HfApi()

        def upload_model(self,local_path,remote_name):
            upload_folder(
                folder_path=local_path,
                path_in_repo=remote_name,
                repo_id=self.repo_id,
                token=self.token
            )
        
        def download_model(self, remote_name, local_dir):
            return Path(
                snapshot_download(
                    repo_id=self.repo_id,
                    # allows us to download all files from the remote_name folder 
                    allow_patterns=[f"{remote_name}/**"],

                    #allows to replicate the file structure from the hub into the local folder,
                    # when local_dir is used then cache_dir will not be used and a .cache/huggingface/ folder will be created in the root of local_dir to store metadata related to the downloaded files
                    local_dir=local_dir,
                    token=self.token
                )
            )
        
        def download_file(self,remote_path,local_dir):
            return Path(
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename=remote_path,
                    local_dir=local_dir,
                    token=self.token
                )
            )