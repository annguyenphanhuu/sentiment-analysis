import pandas as pd
import re
from underthesea import sent_tokenize
from underthesea import text_normalize
from underthesea import word_tokenize

import torch
import pandas as pd
import numpy as np
from transformers import RobertaForSequenceClassification, AutoTokenizer

def is_vietnamese(sentence):
    try:
        # Phát hiện ngôn ngữ của câu
        language = detect(sentence)
    except:
        return False

    # Kiểm tra xem ngôn ngữ có phải là tiếng Việt hay không
    if language == 'vi':
        return True
    else:
        return False

def remove_non_vietnamese(df):
    # Sử dụng hàm is_vietnamese đã được định nghĩa trước đó
    df = df[df['content'].apply(is_vietnamese)]
    return df


def remove_special_characters(df):
    df['content'] = df['content'].astype(str)
    df['content'] = df['content'].apply(lambda string: re.sub(r'[^\w.,]', ' ', string))
    return df


def remove_element_HTML(df):
    df['content'] = df['content'].replace({" a ": " ", " href ": " ", " br ": " "}, regex=True)
    return df

def replace_teencode(df, teencode_path):
    # Đọc file teencode
    with open(teencode_path, 'r', encoding='utf-8') as f:
        teencode_dict = dict(line.strip().split('\t') for line in f)

    # Thay thế từ viết tắt trong df
    df['content'] = df['content'].apply(lambda x: ' '.join(teencode_dict.get(word, word) for word in x.split()))

    return df


def remove_nan(df):
    df = df.dropna(subset=['content'])
    return df

#Sentence segmentation
def sentence_segmentation(df):
    df['content'] = df['content'].apply(sent_tokenize)
    df = df.explode('content')
    df = remove_nan(df)
    return df

#Text normalization
def text_normalization(df):
    df['content'] = df['content'].apply(text_normalize)
    return df

#Word Segmentation
def word_segmentation(df):
    df['content'] = df['content'].apply(lambda x: word_tokenize(x, format="text"))
    return df

def remove_links(df):
    df['content'] = df['content'].apply(lambda x: re.sub(r'http\S+|www.\S+', '', str(x), flags=re.MULTILINE))
    return df

teencode_path = "./Preprocess/teencode4.txt"
def preprocess(df):
    df = remove_nan(df)
    #df = remove_non_vietnamese(df)
    df = remove_links(df)
    df = remove_special_characters(df)
    df = replace_teencode(df, teencode_path)
    df = remove_element_HTML(df) 
    df = sentence_segmentation(df)
    df = text_normalization(df)
    df = word_segmentation(df)
    return df


model = RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")

tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)

def SentimentLabel(sentence):
    input_ids = torch.tensor([tokenizer.encode(sentence)])

    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits.softmax(dim=-1).tolist()
        return(np.argmax(logits[0]))

# Output:
    # [[0.002, 0.988, 0.01]]
    #     ^      ^      ^
    #    NEG    POS    NEU

def apply_sentiment_label(df):
    df = df[df['content'].str.len() <= 256]
    for index, row in df.iloc[:].iterrows():
      df.loc[index, 'label'] = SentimentLabel(row['content'])
    return df

df = pd.read_csv("./DATA/YouTube_crawlData/YoutubeComments.csv",  encoding='latin-1')
print("Preprocessing...")
df = preprocess(df)


df.to_csv("./DATA/YouTube_crawlData/YoutubeComments(Processed).csv")