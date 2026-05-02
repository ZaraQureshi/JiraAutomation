from pathlib import Path
from typing import List, Optional

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, pipeline
from sentence_transformers import SentenceTransformer, datasets
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from datasets import Dataset, ClassLabel, DatasetDict
import numpy as np
import joblib
import pandas as pd
import torch
import json

class PriorityPredictor:
    def __init__(self,model_path=None):
        self.model_path=model_path
        self.num_labels:int=0
        self.label_map:dict={}
        self.tokenizer=None
        self.device=0 if torch.cuda.is_available() else -1

    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        return {"accuracy": accuracy_score(labels, predictions),
                "f1": f1_score(labels, predictions, average="weighted")}
    
    def load_tokenizer(self):
        tokenizer=DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.tokenizer=tokenizer
        return self.tokenizer

    def tokenize_data(self, sample, save_path, max_length=128):
        if self.tokenizer is None:
            self.tokenizer=self.load_tokenizer()
            print(f"self.tokenizer:{self.tokenizer}")

        dataset=Dataset.from_pandas(pd.DataFrame({
            'text': sample['text_combined'],                                                    
            'label': sample['priority.id'] 
        }))

        

         # Label mapping                                                                                                 
        unique_ids = sorted(set(dataset['label'])) 
        print(f"dataset['label']:{dataset['label']}")
   
        print(f"unique_ids:{unique_ids}")

        # the labels are 1,2,3,4,5. we are mapping them to corresponding 0,1,2,3,4. label_map={1: 0, 2: 1, 3: 2}
        # i is the index and old is the original value of the unique_ids
        # we are doing this because the model's final layer (logits) starts at 0

        # without this setp, if the priority.ids were [10,20,30], the model would create 31 neurons in the output layer, with most of them being useless and resulting into value error
                             
        self.label_map = {old: i for i, old in enumerate(unique_ids)}     

        # when we load from disk, we have the remapped labels but we have lost the mapping back to the original priority.ids
        # therefore, creating a json to save the original mapping
        label_map_path = Path(save_path).parent / "label_map.json"
        with open(label_map_path, 'w') as f:
            json.dump(self.label_map, f)

        self.num_labels = len(unique_ids)                                                                               

        dataset = dataset.map(lambda x: {'label': self.label_map[x['label']]})                                          
        dataset = dataset.cast_column("label", ClassLabel(num_classes=self.num_labels))                                 
        
       
        # Tokenize                                                                                                      
        def tokenize_fn(batch):                                                                                         
            return self.tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)          

        tokenized_dataset = dataset.map(tokenize_fn, batched=True) 

        # Split data after tokenizing, so that tokenization is not done on both the splits separately because it is inefficient and causes data leakage.                                                                                                  
        split = tokenized_dataset.train_test_split(test_size=0.2)                                                       
        print(f"split dataset: {split}")                                                   
        # tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])  

        #save tokenized data                       
        Path(save_path).mkdir(parents=True, exist_ok=True)                                                              
        split.save_to_disk(save_path)                                                                       
        print(f"Tokenized data saved to: {save_path}")                                                                                                                       
        return split                     


    def train(self, tokenized_dataset ,save_path):
        # Extract num_labels from dataset if not already set
        if self.num_labels == 0 or self.num_labels is None:
          unique_labels = set(tokenized_dataset["train"]["label"])
          self.num_labels = len(unique_labels)
          print(f"Extracted num_labels={self.num_labels} from dataset")

        print(f"self.num_labels: {self.num_labels}")                                                                                                              
        # Load model                                                                                                    
        model = DistilBertForSequenceClassification.from_pretrained(                                                    
        "distilbert-base-uncased", num_labels=self.num_labels                                                           
        ) 
     
       
     
        
        args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=1,
            per_device_train_batch_size=16,
            max_steps=16,
            report_to="none"
        )
        print(f"Tokenized_data:{tokenized_dataset}")
        trainer = Trainer(
            model=model, args=args, train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"], compute_metrics=self.compute_metrics
        )
        

        trainer.train()
        trainer.save_model(save_path)
        if self.tokenizer:
           self.tokenizer.save_pretrained(save_path)
        
        return {"path": save_path}
    
    def training_process(self,sample,model_path,token_path):
        print(f"model_path:{model_path}")
        print(f"token_path:{token_path}")
        
        if Path(token_path).exists():
            # Load pre-tokenized data from disk
            tokenized_dataset = DatasetDict.load_from_disk(token_path)

             # Extract num_labels from loaded data
            unique_labels = set(tokenized_dataset["train"]["label"])
            self.num_labels = len(unique_labels)

            #loading label_map from json to retain the original mapping of teh priority.ids
            label_map_path = Path(token_path).parent / "label_map.json"
            if label_map_path.exists():
                with open(label_map_path, 'r') as f:
                    self.label_map = json.load(f)
            print(f"Loaded tokenized data from: {token_path}")
        else:
            # Tokenize and save
            tokenized_dataset = self.tokenize_data(sample, token_path)

        path=self.train(tokenized_dataset,model_path)
        return {"path": path}

    def predict(self, text):
        classifier = pipeline("text-classification", model=self.model_path+"/priority_model_v1", device=self.device)
        return classifier(text)[0]


class SimilarTicketFinder:
    def __init__(self, save_path = None, model_name = 'all-MiniLM-L6-v2'):                                   
        self.model_name = model_name                                                                                    
        self.model = SentenceTransformer(model_name)                                                                    
        self.save_path = save_path                                                                                    
        self.embeddings: Optional[np.ndarray] = None

    def compute_and_save_embeddings(self,df):
        self.embeddings=self.model.encode(df['text_combined'].tolist(),show_progress_bar=True)
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)                                                       
        joblib.dump(self.embeddings, self.save_path)  
        # self.save_path=save_path
        print(f"Embeddings saved to: {self.save_path}")
        return self.embeddings
    
    def load_embeddings(self,load_path):
        self.embeddings=joblib.load(load_path)
        return self.embeddings
    
    def find(self,text,df,embeddings=None,top_k=5):
        if embeddings is None:
            if self.embeddings is None:
                embeddings=self.load_embeddings(self.save_path)
        query_vec=self.model.encode([text])
        sims=cosine_similarity(query_vec,embeddings).flatten()
        idx=sims.argsort()[-top_k:][::-1]                                                                             
        results = df.iloc[idx].copy()                                                                        
        results['similarity_score'] = sims[idx]                                                                         
        return results                     
    
class DeveloperMoodChecker:
    def __init__(self):
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.device = 0 if torch.cuda.is_available() else -1
        # Lazy load pipeline
        self.analyzer = None

    def _load_analyzer(self):                                                                                           
        # Lazy load the sentiment analyzer.                                                                       
        if self.analyzer is None:                                                                                       
            self.analyzer = pipeline("sentiment-analysis", model=self.model_name, device=self.device)                   
        return self.analyzer  

    def analyze(self,comments_df,developer,max_comments=10):
        analyzer=self._load_analyzer()
        dev_comments = comments_df[                                                                                     
                  comments_df['comment.author'] == developer]['comment.body'].dropna().tolist()                                                                             
        
        if not dev_comments:
            return {"status": "No data"}
         # Batch inference (much faster)
        results = analyzer(dev_comments[:max_comments], truncation=True)
        # map scores: positive (1) - negative (-1)
        scores = [1 if r['label'] == 'positive' else (-1 if r['label'] == 'negative' else 0) for r in results]
        avg = np.mean(scores)

        return {"avg_score": avg, "label": "Positive" if avg > 0.1 else "Neutral/Stressed"}
