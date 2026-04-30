from src.config import get_settings
from src.data import DataLoader
from src.models import DeveloperMoodChecker, PriorityPredictor, SimilarTicketFinder
from src.storage import HFStorage


class JiraMLService:
    def __init__(self,issues_df,comments_df):
        self.settings=get_settings()
        self.storage=HFStorage()
        self.issues_df = issues_df
        self.comments_df = comments_df
        self.predictor = None
        self.finder = None

    def initialize(self):
        loader=DataLoader(self.settings.data_dir/"filtered_issues.csv",self.settings.data_dir/"filtered_comments.csv")
        self.issues_df=loader.load_issues()
        self.comments_df = loader.load_comments()

        model_path=self.storage.download_model(
            "priority_model",self.settings.model_dir
        )
        embeddings_path = self.storage.download_file(
            "embeddings",
            str(self.settings.model_dir)
        )

        self.predictor = PriorityPredictor(model_path=str(model_path))
        self.finder = SimilarTicketFinder(save_path=str(embeddings_path))
        return self
    
    def analyze(self, summary, description):
        full_text = f"{summary} {description}"
        priority = self.predictor.predict(full_text)
        similar = self.finder.find(full_text, self.issues_df)
        top_dev = similar.iloc[0]['assignee'] if len(similar) > 0 else "unknown"
        mood = DeveloperMoodChecker().analyze(self.comments_df, top_dev)
        
        return {
            "predicted_priority": priority,
            "recommended_dev": top_dev,
            "dev_mood": mood,
            "similar_tickets": similar[['key', 'similarity_score']].to_dict('records') if len(similar) > 0 else []
        }
# Global service instance (lazy loaded)
_service = None

def get_service() -> JiraMLService:
    global _service
    if _service is None:
        _service = JiraMLService()
        _service.initialize()
    return _service