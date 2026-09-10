"""FinetunedLLM: SciBERT encoder + classification head."""
import os

import torch
# [nb cell 1, line 5]
from pathlib import Path
# [nb cell 1, line 14]
import json
# [nb cell 72, lines 1-2]
import torch.nn as nn
from transformers import BertModel
# [nb cell 81, line 1]
import torch.nn.functional as F


# [nb cell 82, lines 1-50]
class FinetunedLLM(nn.Module):
    def __init__(self,llm,embedding_dim,num_classes,dropout_p):
        super().__init__()
        self.llm=llm
        self.embedding_dim=embedding_dim
        self.num_classes=num_classes
        self.dropout_p=dropout_p
        self.dropout=nn.Dropout(dropout_p)
        self.fc1=nn.Linear(embedding_dim,num_classes)

    def forward(self,batch):
        ids,mask=batch['ids'],batch['masks']
        _,pool=self.llm(input_ids=ids,attention_mask=mask)
        logits=self.dropout(pool)
        logits=self.fc1(logits)
        return logits
    
    @torch.inference_mode()
    def predict(self,batch):
        self.eval()
        logits=self(batch)
        y_pred=torch.argmax(logits,dim=1).cpu().numpy()
        return y_pred

    @torch.inference_mode()
    def predict_proba(self,batch):
        self.eval()
        logits=self(batch)
        y_probs=F.softmax(logits,dim=1).cpu().numpy()
        return y_probs

    def save(self,dp):
        with open(Path(dp,"args.json"),'w') as fp:
            content={
                'embedding_dim':self.embedding_dim,
                'num_classes':self.num_classes,
                'dropout_p':self.dropout_p
            }
            json.dump(content,fp=fp,indent=4,sort_keys=False)
        torch.save(self.state_dict(),os.path.join(dp,'model.pt'))

    @classmethod
    def load(cls,args_fp,dict_fp):
        with open(args_fp,'r') as fp:
            kwargs=json.load(fp=fp)
        llm=BertModel.from_pretrained("allenai/scibert_scivocab_uncased",return_dict=False)
        model=cls(llm=llm,**kwargs)
        model.load_state_dict(torch.load(dict_fp,map_location=torch.device('cpu')))
        return model
