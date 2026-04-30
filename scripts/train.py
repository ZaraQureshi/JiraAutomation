from src.config import get_settings
from src.data import DataLoader
from src.models import PriorityPredictor, SimilarTicketFinder
from src.storage import HFStorage


def main():
    settings=get_settings()

    # Load data
    loader = DataLoader(
        str(settings.data_dir / "issues.csv"),
        str(settings.data_dir / "comments.csv")
    )
    issues = loader.load_issues()
    # Train priority model
    print("Training priority model...")
    predictor = PriorityPredictor()
    model_path = str(settings.model_dir / "priority_model_v1")
    token_path=str(settings.token_dir)
    meta = predictor.training_process(issues, model_path,token_path)
    print(f"Model trained: {model_path}")

    # Build similarity index
    print("Building similarity index...")
    finder = SimilarTicketFinder()
    embeddings_path = str(settings.model_dir / "similarity_index.joblib")
    finder.save_path = embeddings_path
    finder.compute_and_save_embeddings(issues)
    print(f"Embeddings saved: {embeddings_path}")
    
    # Upload to HF Hub
    print("Uploading to HF Hub...")
    storage = HFStorage(settings.hf_repo_id, settings.hf_token)
    storage.upload_model(model_path, "priority_model_v1/")
    storage.upload_file(embeddings_path, "embeddings/similarity_index.joblib")
    print("Upload complete!")

if __name__ == "__main__":
    main()