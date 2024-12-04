# 得到和查询相似度最高的k个文本
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from glob import glob
from tqdm import tqdm

def insert_value_at_tensor(tensor, value, position):
    if position < 0 or position >= len(tensor):
        raise ValueError("Position should between 0 and tensor length-1")

    value_tensor = torch.tensor([value], dtype=tensor.dtype)
    new_tensor = torch.cat([tensor[:position], value_tensor, tensor[position:]])
    new_tensor = new_tensor[:-1]

    return new_tensor

def insert_value_at_list(lst, value, position):
    if position < 0 or position >= len(lst):
        raise ValueError("Position should between 0 and lst length-1")

    lst.insert(position, value)
    lst.pop()

    return lst


model = SentenceTransformer('all-MiniLM-L6-v2')
test = pd.read_csv('test.csv')

embs = []
display = True
for _, row in test.iterrows():
    sentences = [
        # 'query: '+  # E5 系列模型要求
        row.prompt
        + " "
        + row.A
        + " "
        + row.B
        + " "
        + row.C
        + " "
        + row.D
        + " "
        + row.E
    ]

    embeddings = torch.Tensor(model.encode(sentences, normalize_embeddings=True)) #[1,384]
    if display:
        print(torch.tensor(embeddings).shape)
        display = False
    embs.append(embeddings)

query_embeddings = torch.Tensor(np.stack(embs)).squeeze(1) #[200,1,384] -> [200,384]
print(query_embeddings.shape)


TOPK = 5
files_wiki = sorted(glob('wiki/wiki_*.parquet'))
files_np = sorted(glob('file_np/wiki_*.npy'))
files = [(x,y) for x, y in zip(files_wiki, files_np)]

max_vals = torch.full((len(test), TOPK), -float("inf")) #[200,5]
max_texts = [[None] * TOPK for _ in range(len(test))]
# 相似度比较 取topk
for file_wiki, file_np in tqdm(files, desc='compare'):
    content_embeddings = torch.Tensor(np.load(file_np)) # [3063, 384]
    sim_scores = torch.matmul(query_embeddings, content_embeddings.transpose(0,1)) # [200,3063] 相似度矩阵
    values, indices = torch.topk(sim_scores, TOPK, dim=1) #[200,5],[200,5]

    file = pd.read_parquet(file_wiki)

    for i in range(len(test)): # 逐行比较
        for new in range(TOPK):
            if values[i][new] < max_vals[i][TOPK-1]:
                break
            for old in range(TOPK):
                if values[i][new] > max_vals[i][old]:
                    max_vals[i] = insert_value_at_tensor(max_vals[i], values[i][new], old)
                    max_texts[i] = insert_value_at_list(max_texts[i], file.iloc[indices[i][new].item()].content, old)
                    break


# context_v2
test['context_v2'] = [
    'Context4: '
    + x[4]
    + '\n###\n'
    + "Context 3: "
    + x[3]
    + "\n###\n"
    + "Context 2: "
    + x[2]
    + "\n###\n"
    + "Context 1: "
    + x[1]
    + "\n###\n"
    + "Context 0: "
    + x[0]
    for x in max_texts
]

test.to_parquet('test_raw.parquet', index=False)
