#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import pandas as pd
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
#dataset
from typing import List
from sklearn.model_selection import KFold, GroupKFold
import random
import os
import numpy as np
from datasets import Dataset
import gc
import torch


# In[3]:


train = pd.read_csv('/kaggle/input/kaggle-llm-science-exam/train.csv')
test = pd.read_csv('/kaggle/input/kaggle-llm-science-exam/test.csv')


# In[4]:


class_num_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
train['answer'] = train['answer'].apply(lambda x: class_num_map[x])
train.head()


# In[5]:


class CFG:
    model_name='/kaggle/input/microsoft-deberta-v3-base'
    output_dir='/kaggle/working/microsoft-deberta-v3-base-finetuing'
    batch_size=8
    learning_rate=1e-5
    weight_decay=1e-8
    hidden_dropout_prob=0. #
    attention_probs_dropout_prob=0. #
    num_train_epochs=2
    save_steps=200
    max_length=1600
    folds=[0,1,2,3]


# In[6]:


# seed
def seed_everything(seed:int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(seed=42)    


# In[7]:


# fold
gfk = GroupKFold(n_splits=len(CFG.folds))
for i, (_, val_index) in enumerate(gfk.split(train, groups=train['prompt'])):
    train.loc[val_index,'fold'] = i
    


# In[8]:


#
def map_at_k(predictions, targets, k=3):
    """
    Computes the Mean Average Precision at K (MAP@K).
    
    Args:
        predictions (list of list): A list of lists, where each sublist contains the predicted labels in ranked order.
        targets (list of list): A list of lists, where each sublist contains the ground truth labels.
        k (int): The cutoff for evaluation (default is 3).

    Returns:
        float: The MAP@K score.
    """
    assert len(predictions) == len(targets), "Predictions and targets must have the same length"
    
    def apk(actual, predicted, k):
        """
        Computes the Average Precision at K for a single instance.
        """
        if len(predicted) > k:
            predicted = predicted[:k]
        
        score = 0.0
        num_hits = 0.0
        
        for i, pred in enumerate(predicted):
            if pred in actual and pred not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        return score / min(len(actual), k) if actual else 0.0

    # Compute the MAP@K
    mapk_score = 0.0
    for pred, true in zip(predictions, targets):
        mapk_score += apk(true, pred, k)
    
    return mapk_score / len(predictions)

    
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Get top-k predictions (e.g., top 3 for MAP@3)
    top_k_preds = predictions.argsort(axis=-1)[:, -3:][:, ::-1]  # Top-3 predictions
    return {"map@3": map_at_k(top_k_preds, labels)}    


# In[9]:


class ExamClassifier:
    def __init__(self, model_name:str, input_text_cols:List[str],output_dir:str,target_col:str,max_length:int,
                hidden_dropout_prob:float,
                attention_probs_dropout_prob:float):
        self.input_col = 'input'
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(self.model_name)
        # config
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.input_text_cols = input_text_cols
        self.output_dir = output_dir
        self.target_col = target_col
        self.max_length = max_length
        self.model_config = AutoConfig.from_pretrained(self.model_name)
        self.model_config.update({
            "hidden_dropout_prob":hidden_dropout_prob,
            "attention_probs_dropout_prob":attention_probs_dropout_prob,
            "num_labels":5,
        })
        self.data_collator=DataCollatorWithPadding( 
            tokenizer=self.tokenizer
        ) #
        
    #    
    def concatenate_with_sep_token(self, row:pd.Series):
        sep = " " + self.tokenizer.sep_token + " " #
        return sep.join(row[self.input_text_cols])
        
    def tokenize_function(self, row:pd.DataFrame):
        labels = row[self.target_col]
        tokenized = self.tokenizer(row[self.input_col],
                                  padding='longest',
                                  truncation=True,
                                  max_length=self.max_length)
        
        return {
            **tokenized,
            'labels': labels,
        }
    
    def tokenize_function_test(self, row:pd.DataFrame):
        tokenized = self.tokenizer(row[self.input_col],
                                  padding='longest',
                                  truncation=True,
                                  max_length=self.max_length)
        
        return tokenized
    
    
    def train(self, train_df:pd.DataFrame,val_df:pd.DataFrame,batch_size:int,learning_rate:float,
             weight_decay:float,num_train_epochs:float,save_steps:int):
        train_df[self.input_col] = train_df.apply(self.concatenate_with_sep_token, axis=1)
        val_df[self.input_col] = val_df.apply(self.concatenate_with_sep_token, axis=1)
        
        train_df = train_df[[self.input_col, self.target_col]]
        val_df = val_df[[self.input_col, self.target_col]]
        # transfer dataset
        train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
        val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
        
        train_tokenized_datasets = train_dataset.map(self.tokenize_function, batched=False)
        val_tokenized_datasets = val_dataset.map(self.tokenize_function, batched=False)
        
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.model_config)
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            load_best_model_at_end=True,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            gradient_checkpointing=True,
            report_to='none',
            greater_is_better=True,
            save_strategy='steps',
            evaluation_strategy='steps',
            eval_steps=save_steps,
            save_steps=save_steps,
            metric_for_best_model='map@3',#
            save_total_limit=1,
            fp16=True,
            auto_find_batch_size=True,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized_datasets,
            eval_dataset=val_tokenized_datasets,
            tokenizer=self.tokenizer,#
            compute_metrics=compute_metrics,#
            data_collator=self.data_collator,
        )
        trainer.train()
        
        model.save_pretrained(self.output_dir)
        
        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
    def predict(self, test_df:pd.DataFrame, fold:int,batch_size:int):
        # transfer dataset
        test_df[self.input_col] = test_df.apply(self.concatenate_with_sep_token, axis=1)
        test_dataset = Dataset.from_pandas(test_df, preserve_index=False)
        test_tokenized_dataset = test_dataset.map(self.tokenize_function_test, batched=False)
        
        model = AutoModelForSequenceClassification.from_pretrained(self.output_dir)
        model.eval()
        
        test_args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=batch_size, #
            dataloader_drop_last=False,
            fp16=True,
            auto_find_batch_size=True,
            report_to='none',
        )
        test = Trainer(
            model=model,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            args=test_args
        )
        output = test.predict(test_tokenized_dataset) #
        # print('predict output:', output)
        logits = output.predictions
        probs = torch.nn.functional.softmax(torch.tensor(logits),dim=-1)
        # pred格式
        
        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        return probs


# In[10]:


def train_by_fold(train:pd.DataFrame, folds:List[int], input_text_cols:List[str],target_col:str): #
    for fold in folds:
        train_df = train[train['fold'] != fold] #
        val_df = train[train['fold'] == fold]
        
        output_dir = f'{CFG.output_dir}/fold_{fold}'
        ec = ExamClassifier(model_name=CFG.model_name, input_text_cols=input_text_cols,output_dir=output_dir,
                           target_col=target_col,max_length=CFG.max_length,hidden_dropout_prob=CFG.hidden_dropout_prob,
    attention_probs_dropout_prob=CFG.attention_probs_dropout_prob,)
        ec.train(train_df=train_df,val_df=val_df,batch_size=CFG.batch_size,learning_rate=CFG.learning_rate,
             weight_decay=CFG.weight_decay,num_train_epochs=CFG.num_train_epochs,save_steps=CFG.save_steps)


# In[11]:


def predict(test:pd.DataFrame, folds:List[int],input_text_cols:List[str],target_col:str,batch_size:int):
#     probs = torch.empty(test.shape[0] * len(folds), num_classes) #folds*test行，5列
    probs = []
    for fold in folds:
        output_dir = f'{CFG.output_dir}/fold_{fold}'
        ec = ExamClassifier(model_name=CFG.model_name, input_text_cols=input_text_cols,output_dir=output_dir,
                           target_col=target_col,max_length=CFG.max_length,hidden_dropout_prob=CFG.hidden_dropout_prob,
    attention_probs_dropout_prob=CFG.attention_probs_dropout_prob)
        prob = ec.predict(test_df=test, fold=fold,batch_size=batch_size)
        probs.append(prob)
        
    # mean probs,softmax
    stacked = torch.stack(probs)
    mean_probs = stacked.mean(dim=0)
    
    # softmax_probs = torch.nn.functional.softmax(mean_probs)
    _,top3_indices = torch.topk(mean_probs, k=3, dim=-1)
    
    return top3_indices


# In[12]:


target_col = 'answer'
input_text_cols = ['prompt','A','B','C','D','E']
class_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}


# In[13]:


train_by_fold(train, CFG.folds,input_text_cols,target_col)


# In[14]:


top3_indices = predict(test, CFG.folds,input_text_cols,target_col, CFG.batch_size)
#map
top3_classes = [[class_map[idx.item()] for idx in row] for row in top3_indices]


# In[15]:


target = [" ".join(row) for row in top3_classes]
target_df = pd.DataFrame(target, columns=['prediction'])
submission = pd.concat([test['id'], target_df], axis=1)
submission.to_csv('/kaggle/working/submission.csv', index=False)


# In[16]:


submission.head()

