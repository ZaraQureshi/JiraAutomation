import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self, issues_path, comments_path):
        self.issues_path=issues_path
        self.comments_path=comments_path
    
    def load_issues(self):
        df = pd.read_csv(self.issues_path, dtype={'customfield_12310921': 'object'})
        return self._clean_issues(df)
    
    def load_comments(self):
        return pd.read_csv(self.comments_path)

    def _clean_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=['summary'])
        df['description'] = df['description'].fillna(df['summary'])
        df['text_combined'] = df['summary'] + " " + df['description']
        
        required = ['key', 'text_combined', 'issuetype.name', 'project.key',
                    'project.name', 'reporter', 'assignee', 'status.name', 'priority.id']
        cols = [c for c in required if c in df.columns]
        
        counts = df["priority.id"].value_counts()
        mainstream = counts.head(5).index.tolist()
        df = df[df["priority.id"].isin(mainstream)]
        
        return df[cols].copy()