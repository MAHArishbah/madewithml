"""Inspect SciBERT: parameters, buffers, state_dict, a forward pass, the FinetunedLLM head.

Run from the project root:
    python diagnostics/model_inspection.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets `import madewithml` work from any folder

import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer

from madewithml.config import DATASET_LOC
from madewithml.models import FinetunedLLM


def main():
    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")  # [added] notebook defined it in EDA cell 25
    num_classes = pd.read_csv(DATASET_LOC).tag.nunique()  # [added] notebook defined it in EDA cell 21

    # [nb cell 73, lines 1-3]
    #PreTrained LLM
    llm=BertModel.from_pretrained("allenai/scibert_scivocab_uncased",return_dict=False) #return dict for controlling output 
    embedding_dim=llm.config.hidden_size

    # [nb cell 74, line 1]
    print(llm)

    # [nb cell 75, lines 1-2]
    print(list(llm.state_dict().keys()))


    # [nb cell 76, lines 1-10]
    # for name , p in llm.named_parameters():
    #     print(f'{name} requires grad {p.requires_grad} {tuple(p.shape)}')

    total=sum(p.numel() for p in llm.parameters())
    learnt=sum(p.numel() for p in llm.parameters() if p.requires_grad)
    learnable=learnt/total
    print(f'total {total} learnable {100*learnable}')

    for n ,_ in llm.named_buffers():
        print(n)

    # [nb cell 77, lines 1-2]
    x=llm.named_parameters()
    print(list(x))

    # [nb cell 78, lines 1-2]
    y=list(llm.parameters())
    print(y)

    # [nb cell 79, lines 1-2]
    print(llm.state_dict())


    # [nb cell 80, lines 1-6]
    # Sample sanity 
    text = "Transfer learning with transformers for text classification."
    batch = tokenizer([text], return_tensors="pt", padding="longest")
    # batch = {k:torch.tensor(v) for k,v in batch.items()}  # convert to torch tensors
    seq, pool = llm(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    print(np.shape(seq), np.shape(pool))

    # [nb cell 83, lines 1-4]
    #intitialize model
    model=FinetunedLLM(llm=llm,dropout_p=.5,embedding_dim=embedding_dim,num_classes=num_classes)
    print(list(model.named_parameters()))


    # [nb cell 84, line 1]
    print(llm.encoder.layer[1].attention.self.key.weight)

    # [nb cell 102, lines 1-2]
    # The driver still holds two SciBERT copies from the exploration cells above (~0.9 GB).
    # train_loop_per_worker loads its own model, so dropping them to save ram and prevent OOM 
    # [nb cell 102, line 4]
    del llm, model


if __name__ == "__main__":
    main()
