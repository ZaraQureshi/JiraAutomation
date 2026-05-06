from src.config import get_settings
from src.data import DataLoader
from src.models import PriorityPredictor, SimilarTicketFinder
from src.storage import HFStorage

def train_model(issues,model_path,token_path):
    
    # Train priority model
    print("Training priority model...")
    predictor = PriorityPredictor()
    
    meta = predictor.training_process(issues, model_path,token_path)
    print(f"Model trained: {model_path}")

def create_embeddings(settings,issues):
     # Build similarity index
    print("Building similarity index...")
    finder = SimilarTicketFinder()
    embeddings_path = str(settings.model_dir / "similarity_index.joblib")
    finder.save_path = embeddings_path
    finder.compute_and_save_embeddings(issues)
    print(f"Embeddings saved: {embeddings_path}")

def upload_HF(settings, model_path, embeddings_path):
    print("Uploading to HF Hub...")
    storage = HFStorage(settings.hf_repo_id, settings.hf_token)
    storage.upload_model(model_path, "priority_model_v1/")
    storage.upload_file(embeddings_path, "similarity_index.joblib")
    print("Upload complete!")

def main():
    settings=get_settings()

    # Load data
    loader = DataLoader(settings.dataset_repo_id,settings.hf_token)
    issues = loader.load_issues()
    embeddings_path = str(settings.model_dir / "similarity_index.joblib")
    model_path = str(settings.model_dir / "priority_model_v1")
    token_path=str(settings.token_dir)

    # Train priority model
    train_model(issues,model_path,token_path)

    # Build similarity index
    create_embeddings(settings,issues)
    
    # Upload to HF Hub
    upload_HF(settings, model_path, embeddings_path)

if __name__ == "__main__":
    main()