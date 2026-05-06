import pandas as pd
from pathlib import Path
from datasets import load_dataset
class DataLoader:
    def __init__(self, repo_id, token):
        self.repo_id = repo_id
        self.token = token
        
    def load_issues(self):
        # Stream from cloud directly to memory
        ds = load_dataset(self.repo_id, data_files="issues.csv", split="train",
                          token=self.token, streaming=False)
        df = ds.to_pandas()
        # Ensure your custom field is treated as an object
        if 'customfield_12310921' in df.columns:
            df['customfield_12310921'] = df['customfield_12310921'].astype(object)
        return self._clean_issues(df)
    
    def load_comments(self):
        ds = load_dataset(self.repo_id, data_files="comments.csv", split="train",
                          token=self.token, streaming=False)
        return ds.to_pandas()

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