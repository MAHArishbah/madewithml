"""Zero-/few-shot baselines with Claude on the holdout set.

Calls the Anthropic API: needs ANTHROPIC_API_KEY and costs tokens.
Run from the project root:
    python diagnostics/llm_baselines.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets `import madewithml` work from any folder

import json
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# [nb cell 1, lines 2-4]
import anthropic
import time
from tqdm.auto import tqdm
# [nb cell 1, line 19]
from sklearn.metrics import precision_recall_fscore_support,classification_report,confusion_matrix,ConfusionMatrixDisplay

from madewithml.config import HOLDOUT_LOC

# [nb cell 38, line 1] (markdown cell, copied as comments)
# Lets establish baselines with zeroshot and few shot

# [nb cell 39, lines 1-2]
client =anthropic.Anthropic()


# [nb cell 44, lines 1-20]
# function that can predict tags for a given sample
def get_tag(model,system_content="",assistant_content="",user_content=""):
    messages=[]
    if assistant_content:
        messages.append({"role": "assistant", "content": assistant_content})
    messages.append({"role": "user", "content": user_content})
    try:
        response= client.messages.create(
        model=model,
        max_tokens=100,
        system=system_content,
        messages=messages,
        # temperature=0,        
        )
        text_blocks = [block.text for block in response.content if block.type == "text"]
        predicted_tag=text_blocks[0].strip().lower() if text_blocks else None
        return predicted_tag
    except (anthropic.APIError) as e:
        return None
        

# [nb cell 47, lines 1-14]
def get_predictions(inputs,model,system_content,assistant_content=""):
    y_pred=[]
    for item in tqdm(inputs):
        user_content=str(item)
        # print(item)
        predicted_tag=get_tag(model,system_content,assistant_content,
                              user_content)
        while predicted_tag is None:
            time.sleep(30)
            predicted_tag=predicted_tag=get_tag(model,system_content,assistant_content,
                                          user_content)
        y_pred.append(predicted_tag)
    return y_pred


# [nb cell 49, lines 1-8]
def clean_predictions(y_pred,tags,default='other'):
    for i, item in enumerate(y_pred):
        if item not in tags: #hallucination
            y_pred[i]=default
        if item.startswith("'") and item.endswith("'"): #destringify
            y_pred[i]=item[1:-1]
    return y_pred


# [nb cell 50, lines 1-16]
#plot ground truth and predictions
def plot_tag_dist(y_true,y_pred):
    true_tag_freq=dict(Counter(y_true))
    pred_tag_freq=dict(Counter(y_pred))
    df_true=pd.DataFrame({'tag':list(true_tag_freq.keys()),'freq':list(true_tag_freq.values()),'source': 'true'})
    df_pred=pd.DataFrame({'tag':list(pred_tag_freq.keys()),'freq':list(pred_tag_freq.values()),'source': 'pred'})
    df=pd.concat([df_true,df_pred],ignore_index=True)
    
    plt.figure(figsize=(10, 3))
    plt.title("Tag distribution", fontsize=14)
    ax = sns.barplot(x="tag", y="freq", hue="source", data=df)
    ax.set_xticklabels(list(true_tag_freq.keys()), rotation=0, fontsize=8)
    plt.legend()
    plt.show()



# [nb cell 51, lines 1-16]
def evaluate(test_df,model,system_content,tags,assistant_content=''):

    y_test=test_df['tag'].to_list()
    test_samples=test_df[['title','description']].to_dict(orient='records')
    y_pred=get_predictions(
        inputs=test_samples,model=model,
        system_content=system_content,assistant_content=assistant_content
    )
    y_pred=clean_predictions(y_pred=y_pred,tags=tags)


    metrics=precision_recall_fscore_support(y_test,y_pred,average='weighted')
    performance={'precision':metrics[0],'recall':metrics[1],'f1':metrics[2]}
    print(json.dumps(performance,indent=2))
    plot_tag_dist(y_true=y_test,y_pred=y_pred)
    return y_pred,performance


def main():

    # [nb cell 40, lines 1-6]
    # response= client.messages.create(
    #     model='claude-sonnet-5',
    #     max_tokens=100,
    #     messages=[{'role': "user",'content':'tell me a joke'}]
    # )
    # [response.content[0].text][0]

    # [nb cell 41, lines 1-5]
    DATASET_LOC = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
    train_df = pd.read_csv(DATASET_LOC)
    print(train_df.head())
    tags=train_df.tag.unique().tolist()
    print(tags)

    # [nb cell 42, line 2]
    test_df = pd.read_csv(HOLDOUT_LOC)

    # [nb cell 43, line 1]
    print(test_df[['title','tag']][:3])

    # [nb cell 45, lines 1-3]
    # List of dicts] as serving endpoint needs json format 
    samples=test_df[["title","description"]].to_dict(orient='records')[:3]
    print(samples)

    # [nb cell 46, lines 1-16]
    #sanity
    model='claude-sonnet-5'
    system_context = f"""
    You are a NLP prediction service that predicts the label given an input's title and description.
    You must choose between one of the following labels for each input: {tags}.
    Only respond with the label name and nothing else.
    """
    assistant_content=""
    user_content ={'title': 'Diffusion to Vector',
      'description': 'Reference implementation of Diffusion2Vec (Complenet 2018) built on Gensim and NetworkX. '}
    tag=get_tag(model=model,system_content=system_context,assistant_content=assistant_content,
                user_content=str(user_content)) #model expects user input in string format 
    print(tag)




    # [nb cell 48, line 1]
    print(get_predictions(samples,model,system_context))

    # [nb cell 52, lines 1-3]
    #initialise dicts for analysis accros multiple models
    y_pred = {"zero_shot": {}, "few_shot": {}}
    performance = {"zero_shot": {}, "few_shot": {}}

    # [nb cell 53, lines 1-5]
    system_content = f"""
    You are a NLP prediction service that predicts the label given an input's title and description.
    You must choose between one of the following labels for each input: {tags}.
    Only respond with the label name and nothing else.
    """

    # [nb cell 54, lines 1-5]
    method='zero_shot'
    model='claude-sonnet-5'
    y_pred[method][model],performance[method][model]=evaluate(
        test_df=test_df,model=model,system_content=system_content,tags=tags
    )

    # [nb cell 55, lines 1-3]
    #results seems to good , verifying with classification report 
    y_test=test_df['tag'].to_list()
    print(classification_report(y_test,y_pred[method][model], target_names=tags))

    # [nb cell 56, lines 1-6]
    #zeroshot on haiku
    method='zero_shot'
    model='claude-haiku-4-5-20251001'
    y_pred[method][model],performance[method][model]=evaluate(
        test_df=test_df,model=model,system_content=system_content,tags=tags
    )

    # [nb cell 57, lines 1-8]
    #pulling few shot examples from test set would be leakage
    num_samples=2
    additional_context=[]
    cols_to_keep=["title","description","tag"]
    for tag in tags:
        samples=train_df[cols_to_keep][train_df.tag==tag][:num_samples].to_dict(orient='records')
        additional_context.extend(samples)
    print(additional_context)

    # [nb cell 58, lines 1-3]
    # Add assistant context
    assistant_content = f"""Here are some examples with the correct labels: {additional_context}"""
    print (assistant_content)

    # [nb cell 59, lines 1-6]
    # Few-shot on sonnet 5
    method = "few_shot"
    model = "claude-sonnet-5"
    y_pred[method][model], performance[method][model] = evaluate(
        test_df=test_df, model=model, system_content=system_content,
        assistant_content=assistant_content, tags=tags)

    # [nb cell 60, lines 1-6]
    # Few-shot on haiku
    method = "few_shot"
    model = "claude-haiku-4-5-20251001"
    y_pred[method][model], performance[method][model] = evaluate(
        test_df=test_df, model=model, system_content=system_content,
        assistant_content=assistant_content, tags=tags)

    # [nb cell 61, line 1]
    print(classification_report(y_true=y_test,y_pred=y_pred[method][model],target_names=tags))

    # [nb cell 62, lines 1-2]
    cm=confusion_matrix(y_test,y_pred[method][model],labels=tags)
    ConfusionMatrixDisplay(cm,display_labels=tags).plot(xticks_rotation=45,colorbar=False,cmap='coolwarm')

    # [nb cell 63, line 1]
    print(json.dumps(performance,indent=2))

    # [nb cell 64, lines 1-7]
    by_model_context={}
    for context_type, models_data in performance.items():
        for model,metrics in models_data.items():
            key=f'{model}_{context_type}'
            by_model_context[key]=metrics

    print(by_model_context)

    # [nb cell 65, lines 1-23]
    # Plotting the bar chart with metric scores on top of each bar
    models=list(by_model_context.keys())
    metrics=list(by_model_context[models[0]].keys())

    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.2
    x = range(len(models))

    for i, metric in enumerate(metrics):
        metric_values = [by_model_context[model][metric] for model in models]
        ax.bar([pos + width * i for pos in x], metric_values, width, label=metric)
        # Displaying the metric scores on top of each bar
        for pos, val in zip(x, metric_values):
            ax.text(pos + width * i, val, f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks([pos + width for pos in x])
    ax.set_xticklabels(models, rotation=0, ha='center', fontsize=8)
    ax.set_ylabel('Performance')
    ax.set_title('Anthropic Benchmarks')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()

    # [nb cell 66, lines 1-10]
    #another way as data is already in wider format , using pandas
    ax = pd.DataFrame(by_model_context).T.plot.bar(figsize=(10, 4), rot=0, width=0.7,
                                                   title="Anthropic Benchmarks", ylabel="Performance") #outer key becomes column , inner becomes row. so .T
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


    # [nb cell 67, lines 1-17]
    #if was in longer formal then , example
    df = (pd.DataFrame(by_model_context).T         
            .rename_axis("model").reset_index()
            .melt(id_vars="model", var_name="metric", value_name="score"))

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=df, x="model", y="score", hue="metric", errorbar=None, ax=ax)

    for c in ax.containers:                          # one container per metric
        ax.bar_label(c, fmt="%.3f", fontsize=9)

    ax.set(title="Anthropic Benchmarks", xlabel="", ylabel="Performance")
    ax.tick_params(axis="x", labelsize=8)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
