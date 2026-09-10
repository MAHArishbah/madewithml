"""Ray Data checks: cluster resources, stratified split, class mapping, preprocessing, one collated batch.

Run from the project root:
    python diagnostics/ray_data_checks.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets `import madewithml` work from any folder

import numpy as np
import ray
from transformers import BertTokenizer

from madewithml.config import DATASET_LOC, RANDOM_SEED, TEST_SIZE
from madewithml.data import clean_text, preprocess, stratify_split
from madewithml.utils import collate_fn

# [nb cell 29, lines 1-23]
#just for refrence here. not used 
class Preprocessor:
    def __init__(self, class_to_index):
        self.tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")  # once per actor
        self.class_to_index = class_to_index


    def tokenize(self,df):
        encoded_inputs=self.tokenizer(df['text'].tolist(),return_tensors='np',padding='longest',
                             truncation=True,max_length=128)
        return dict(ids=encoded_inputs["input_ids"],masks=encoded_inputs['attention_mask'],targets=np.array(df['tag']))

    def __call__(self, df):
        # same body as your preprocess(), but use self.tokenizer inside tokenize
        df['text']=df['title']+" "+df['description'] #feature eng
        df['text']=df['text'].apply(clean_text) #clean_text
        df = df.drop(columns=["id", "created_on", "title", "description"], errors="ignore")  # clean dataframe
        df = df[["text", "tag"]]  # rearrange columns
        df["tag"] = df["tag"].map(class_to_index)  # label encoding    
        return self.tokenize(df)




def main():
    ray.init(ignore_reinit_error=True)  # [added] train.py owns the notebook's ray.init() (cell 31)

    # [nb cell 32, line 1]
    print(ray.cluster_resources())

    # [nb cell 33, lines 1-3]
    ds=ray.data.read_csv(DATASET_LOC,override_num_blocks=int(ray.cluster_resources().get('CPU',1)))
    ds=ds.random_shuffle(seed=RANDOM_SEED)
    print(ds.take(1))

    # [nb cell 35, lines 1-2]
    train_ds,val_ds=stratify_split(ds,stratify='tag',test_size=TEST_SIZE)
    # train_ds

    # [nb cell 36, lines 1-5]
    #just to check  how to do in ray , earlier class to index can be used here 
    #  tags=train_ds.unique(column='tag')
    # class_to_index = {tag: i for i, tag in enumerate(tags)}
    tags = sorted(train_ds.to_pandas()['tag'].unique()) #to pandas because tag column is small , ray overhead doesnt make sense.
    class_to_index = {tag: i for i, tag in enumerate(tags)}

    # [nb cell 37, lines 1-11]
    # sample_ds_classOne = train_ds.map_batches(
    #   Preprocessor,
    #   fn_constructor_kwargs={"class_to_index": class_to_index},
    #   batch_format="pandas")
    # sample_ds.show(1)

    sample_ds = train_ds.map_batches(
      preprocess,
      fn_kwargs={"class_to_index": class_to_index},
      batch_format="pandas")
    sample_ds.show(1)

    # [nb cell 88, lines 1-3]
    # Sample batch| dataset > shard (1 per worker) > block > rows, batch is collection of rows 
    sample_batch = sample_ds.take_batch(batch_size=128)
    print(collate_fn(batch=sample_batch))


if __name__ == "__main__":
    main()
