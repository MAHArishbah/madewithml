"""EDA on the raw CSV with pandas: shapes, tag balance, plots, word clouds, text cleaning, tokenizer checks.

Run from the project root:
    python diagnostics/eda.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets `import madewithml` work from any folder

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
# [nb cell 1, lines 8-11]
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns;sns.set_theme()
from wordcloud import WordCloud,STOPWORDS

from madewithml.config import DATASET_LOC, RANDOM_SEED, TEST_SIZE
from madewithml.data import clean_text, decode, preprocess, tokenize


def main():
    # [nb cell 3, lines 2-3]
    df = pd.read_csv(DATASET_LOC)
    print(df.head())

    # [nb cell 4, line 1]
    print(df.shape)

    # [nb cell 5, line 1]
    print(df.columns)

    # [nb cell 6, line 1]
    print(df.tag.value_counts())

    # [nb cell 7, line 1]
    train_df, val_df = train_test_split(df,stratify=df.tag,test_size=TEST_SIZE,random_state=RANDOM_SEED)

    # [nb cell 8, line 1]
    print(train_df.shape)

    # [nb cell 9, line 1]
    print(train_df.tag.value_counts())

    # [nb cell 10, line 1]
    print(val_df.shape)

    # [nb cell 11, lines 1-2]
    #verify if startify did its job
    print(val_df.tag.value_counts() * int((1-TEST_SIZE)/TEST_SIZE))

    # [nb cell 12, line 1]
    print(val_df.shape)

    # [nb cell 13, lines 1-5]
    all_tags=Counter(df['tag'])
    print(all_tags.most_common())
    #Equivalent if no counter to be used
    # count=df['tags'].value_counts()
    # tag,tags_counts=count.index,count.values

    # [nb cell 14, lines 1-3]
    plt.figure(figsize=(10,5))
    sns.countplot(df,x='tag',hue='tag',legend=False,palette='Set2',order=df['tag'].value_counts().index)
    plt.tight_layout()

    # [nb cell 15, lines 1-9]
    # Plot tag frequencies #other alternative
    tags, tag_counts=zip(*all_tags.most_common())
    plt.figure(figsize=(10,3))
    ax=sns.barplot(x=list(tags),y=list(tag_counts),hue=tags)
    ax.set_xticklabels(tags,rotation=0)
    plt.title('Tag Distribution',fontsize=14)
    plt.ylabel('# of projects',fontsize=12)
    plt.tight_layout()
    plt.show()

    # [nb cell 16, lines 1-9]
    #most frequent tokens for each tag
    for tag in tags: 
        plt.figure(figsize=(10,3))
        subset=df[df['tag']==tag]
        text=subset.title.values
        cloud=WordCloud(stopwords=STOPWORDS,background_color='black',collocations=False,width=500,height=300).generate(" ".join(text))
        plt.axis('off')
        print(tag)
        plt.imshow(cloud)

    # [nb cell 17, line 1]
    df['text']=df['title']+ " " + df['description']

    # [nb cell 19, lines 1-4]
    original_df=df.copy()
    df['text']=df['text'].apply(clean_text)
    print (f"{original_df.text.values[0]}\n{df.text.values[0]}")


    # [nb cell 20, lines 1-4]
    df=df.drop(columns=['id','created_on','title','description'],errors='ignore')
    df=df.dropna(subset=['tag'])
    df=df[['text','tag']]
    print(df.head())

    # [nb cell 21, lines 1-5]
    #label encoding
    tags=sorted(train_df['tag'].unique().tolist())
    num_classes=len(tags)
    class_to_index={tag:i for i, tag in enumerate(tags)}
    print(class_to_index)

    # [nb cell 22, lines 1-2]
    df['tag']=df['tag'].map(class_to_index)
    print(df.head())

    # [nb cell 24, lines 1-2]
    index_to_class={v:k for k,v in class_to_index.items()}
    print(decode(df.head(1)['tag'].values,index_to_class=index_to_class))

    # [nb cell 25, lines 1-9]
    #bert tokenizer test
    tokenizer=BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    text="Transfer learning with transformers for text classification"
    encoded_inputs=tokenizer([text],return_tensors='np',padding='longest')
    print ("input_ids:", encoded_inputs["input_ids"])
    print ("attention_mask:", encoded_inputs["attention_mask"])
    print (tokenizer.decode(encoded_inputs["input_ids"][0]))



    # [nb cell 27, line 1]
    print(tokenize(df.head(2)))

    # [nb cell 30, line 1]
    print(preprocess(df=train_df,class_to_index=class_to_index))

    plt.show()  # [added] display figures created after the last plt.show() (the word clouds)


if __name__ == "__main__":
    main()
