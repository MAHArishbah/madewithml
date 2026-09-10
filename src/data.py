"""Loading, splitting and preprocessing the dataset."""
# [nb cell 1, line 1]
import ray
# [nb cell 1, lines 6-7]
import pandas as pd
from sklearn.model_selection import train_test_split
# [nb cell 1, lines 12-13]
import nltk
from nltk.corpus import stopwords
# [nb cell 1, lines 15-16]
# from nltk.stem import PorterStemmer
import re
# [nb cell 1, line 18]
from transformers import BertTokenizer
# [nb cell 68, line 3]
from ray.data.preprocessor import Preprocessor
import numpy as np

from madewithml.config import DATASET_LOC, RANDOM_SEED


# [nb cell 70, lines 1-6]
def load_data(num_samples=None):
    ds=ray.data.read_csv(DATASET_LOC,override_num_blocks=int(ray.cluster_resources().get('CPU',1)))
    ds=ds.random_shuffle(seed=RANDOM_SEED)
    ds=ray.data.from_items(ds.take(num_samples)) if num_samples else ds #truncate ray dataset if numsamples valid
    return ds


# [nb cell 34, lines 1-21]
#ray dataset
def stratify_split(ds,stratify,test_size,shuffle=True,seed = 1234):

    def _add_split(df):
        train, test = train_test_split(df,test_size=test_size, shuffle=shuffle, random_state=seed)
        train["_split"] = "train"
        test["_split"] = "test"
        return pd.concat([train, test])

    def _filter_split(df, split):
        return df[df["_split"] == split].drop(columns="_split",errors="ignore")

    
    grouped=ds.groupby(stratify).map_groups(_add_split,batch_format='pandas')
    train_ds=grouped.map_batches(_filter_split,fn_kwargs={"split": "train"}, batch_format="pandas")
    test_ds=grouped.map_batches(_filter_split,fn_kwargs={"split": "test"}, batch_format="pandas")

    train_ds=train_ds.random_shuffle(seed=seed)
    test_ds=test_ds.random_shuffle(seed=seed)

    return train_ds,test_ds


# [nb cell 18, lines 1-17]
#lru cache can be used as well 

def clean_text(text):  
    if not hasattr(clean_text,"_pattern"):
        nltk.download('stopwords',quiet=True)
        sw=stopwords.words('english')
        clean_text._pattern =re.compile(r'\b(' + r'|'.join(sw) + r")\b\s*")
    text=text.lower()    
    text=clean_text._pattern.sub('',text)
    text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)  # add spacing before and after punctuations
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars , puncutations
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends
    text = re.sub(r"http\S+", "", text)  #  remove links

    return text


# [nb cell 26, lines 1-6]
#method to tokenize
def tokenize(batch):
    tokenizer=BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    encoded_inputs=tokenizer(batch['text'].tolist(),return_tensors='np',padding='longest',
                             truncation=True,max_length=128)
    return dict(ids=encoded_inputs["input_ids"],masks=encoded_inputs['attention_mask'],targets=np.array(batch['tag']))


# [nb cell 28, lines 1-10]
# here tokenize is the issue , it gets reloaded every batch , changing this function to class will load once per worker 
def preprocess(df,class_to_index):
    df['text']=df['title']+" "+df['description'] #feature eng
    df['text']=df['text'].apply(clean_text) #clean_text
    df = df.drop(columns=["id", "created_on", "title", "description"], errors="ignore")  # clean dataframe
    df = df[["text", "tag"]]  # rearrange columns
    df["tag"] = df["tag"].map(class_to_index)  # label encoding
    outputs=tokenize(df)
    return outputs


# [nb cell 71, lines 1-10]
# The leading underscores (_fit, _transform_pandas) are the hooks we override,always call the public fit / transform / fit_transform, which do bookkeeping and then call underscore methods
class CustomPreprocessor(Preprocessor):
    def _fit(self, ds):
        tags = sorted(ds.to_pandas()["tag"].unique())
        self.class_to_index = {tag: i for i, tag in enumerate(tags)}
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        return self

    def _transform_pandas(self, batch):
        return preprocess(batch, class_to_index=self.class_to_index)


# [nb cell 23, lines 1-3]
def decode(indices,index_to_class):
    return [index_to_class[index] for index in indices]
