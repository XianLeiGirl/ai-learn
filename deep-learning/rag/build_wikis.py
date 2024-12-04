# wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
# bzip2 -d enwiki-latest-pages-articles.xml.bz2

# python3 WikiExtractor.py -o output enwiki-latest-pages-articles.xml

import os
import pandas as pd

# 定义数据目录（WikiExtractor 输出目录）
data_dir = "output"
index = 0
os.mkdir('wiki')

# 遍历所有提取的文档文件
for dirname, _, filenames in os.walk(data_dir):
    for filename in filenames:
        filepath = os.path.join(dirname, filename)

        data = []
        with open(filepath, 'r', encoding='utf-8',errors='ignore') as file:
            content = file.read()
            docs = content.split('<doc ')
            for doc in docs[1:]:
                lines = doc.split('\n')
                wiki_id = lines[0].split('"')[1]
                title = lines[0].split('title="')[1].split('"')[0]
                paragraphs = lines[1:-2] # 最后2位：</doc> ' '
                for paragraph in paragraphs:
                    paragraph = paragraph.strip()
                    if paragraph:
                        data.append({
                            'wiki_id': wiki_id,
                            'title': title,
                            'text': paragraph,
                            'char_count': len(paragraph)
                        })


        df = pd.DataFrame(data)
        df.to_parquet(f'wiki/wiki_{index:05}.parquet', engine='pyarrow', index=False)

        index += 1             
