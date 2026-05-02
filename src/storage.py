import hashlib

from huggingface_hub import HfApi, ModelCard, ModelCardData, upload_folder,upload_file, snapshot_download, hf_hub_download
from pathlib import Path
from typing import Optional

from sklearn import metrics

class HFStorage:
    def __init__(self,repo_id,token):
        self.repo_id=repo_id
        self.token=token
        self.api=HfApi()

    #make hash of the files to compare with the existing model files and find any differences
    def get_file_hash(self, path):
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def upload_model(self,local_path,remote_name):
        local_file_hash=self.get_file_hash(f"{local_path}/model.safetensors")
        try:
            # Get metadata of the file currently in the repo
            file_info = self.api.get_paths_info(self.repo_id, paths=[f"priority_model_v1/model.safetensors"])
            print(f"File info: {file_info}")
            if file_info and file_info[0].lfs.sha256 == local_file_hash:
                print("No changes detected in model weights. Skipping upload.")
                return
        except Exception:
            print("First time upload or file not found. Proceeding...")

        upload_folder(
            folder_path=local_path,
            path_in_repo=remote_name,
            repo_id=self.repo_id,
            token=self.token
        )
        self.model_card()
        self.model_versioning()

    def upload_file(self,local_path,remote_name):
        local_file_hash=self.get_file_hash(f"{local_path}")
        try:
            # Get metadata of the file currently in the repo
            file_info = self.api.get_paths_info(self.repo_id, paths=[f"similarity_index.joblib"])
            print(f"File info: {file_info}")
            if file_info and file_info[0].lfs.sha256 == local_file_hash:
                print("No changes detected in model weights. Skipping upload.")
                return
        except Exception:
            print("First time upload or file not found. Proceeding...")

        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_name,
            repo_id=self.repo_id,
            token=self.token
        )
        self.model_card()
        self.model_versioning()
       
    
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
    def model_card(self):
        # This stores your tuning results in a machine-readable way
        card_data = ModelCardData(
            language='en',
            license='mit',
            model_name='Jira Priority Predictor',
            eval_results=metrics,  # Accuracy, F1, etc.
            #training_config=params # Learning rate, epochs, etc. to add soon
        )
        
        card = ModelCard.from_template(card_data)
        card.push_to_hub(self.repo_id)
    
    def model_versioning(self):
        # THE VERSIONING: Create a Git Tag
        # This creates a 'snapshot' you can always go back to
        version_tag = f"v_{len(self.api.list_repo_commits(self.repo_id))}" 
        self.api.create_tag(self.repo_id, tag=version_tag, tag_message=f"Tuning run: {version_tag}")

        print(f"Uploaded and tagged as {version_tag}")