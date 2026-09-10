# Cell map: test.ipynb → madewithml_project

Where every line of `test.ipynb` went. Use it to repeat the split by hand on another project.

**How to read the numbers**

- **nb cell**: position of the cell in `test.ipynb`, counting from 1 at the top (markdown cells count too).
- **cell lines**: line numbers inside that cell, counting from 1 (turn on line numbers in VS Code with `L` on a selected cell).
- **file lines**: line numbers in the target file.
- Every copied block in every file starts with a header comment naming its source, e.g. `# [nb cell 69, lines 1-10]`. Search a file for `nb cell 69` to find it.

**Coverage:** 724 code lines in the notebook, 724 placed, 0 missing, 0 duplicated. 33 lines modified, all listed below (mostly notebook display lines such as `df.head()` wrapped in `print()`). The only other changes are added lines (imports, the `train_model()` and `main()` wrappers, headers) and indentation for code moved into those functions.

## Folder layout

```
madewithml_project/
├── madewithml/                 # the package: what training needs
│   ├── __init__.py
│   ├── config.py               # constants: TEST_SIZE, RANDOM_SEED, DATASET_LOC, HOLDOUT_LOC, TRAIN_LOOP_CONFIG, logging
│   ├── utils.py                # set_seeds, get_device, pad_array, collate_fn
│   ├── data.py                 # load_data, stratify_split, clean_text, tokenize, preprocess, CustomPreprocessor, decode
│   ├── models.py               # FinetunedLLM
│   └── train.py                # train_step, eval_step, train_loop_per_worker, train_model
├── diagnostics/                # exploration only, never imported by the package
│   ├── eda.py                  # pandas EDA, plots, word clouds, tokenizer checks
│   ├── ray_data_checks.py      # Ray cluster, split, preprocessing, one collated batch
│   ├── model_inspection.py     # SciBERT parameters/buffers/state_dict, forward pass
│   └── llm_baselines.py        # Claude zero-/few-shot baselines (calls the API)
├── CELL_MAP.md                 # this file
└── cell_map.csv                # same map as a spreadsheet
```

## How to run

From inside `madewithml_project/`, with the same venv as the notebook:

```powershell
$env:PYTHONPATH = (Get-Location).Path     # Ray workers need to import madewithml
python -m madewithml.train                # full run, same settings as the notebook
python -c "from madewithml.train import train_model; train_model(num_samples=100, num_workers_override=1, num_epochs=1)"   # quick check
python diagnostics/eda.py                 # same for ray_data_checks.py, model_inspection.py, llm_baselines.py
```

Rule of thumb when splitting a notebook: functions and classes go to the package module for their stage (`data` / `models` / `utils` / `train`); constants go to `config.py`; top-level cells that *run* training go inside `train_model()`; anything that only displays, plots or checks goes to `diagnostics/`.

Each diagnostics script is laid out like a normal script: imports at the top, the notebook's own functions and classes at module level, every other cell inside `main()` in notebook order, and `if __name__ == "__main__": main()` at the bottom. Lines the notebook only *displayed* (e.g. `df.head()`) are wrapped in `print()`.

## By notebook cell

| nb cell | cell starts with | cell lines | → file | file lines | note |
|---:|---|---|---|---|---|
| 1 | `import ray` | 1 | `madewithml/data.py` | 3 |  |
| 1 | `import ray` | 2-4 | `diagnostics/llm_baselines.py` | 19-21 |  |
| 1 | `import ray` | 5 | `madewithml/models.py` | 6 |  |
| 1 | `import ray` | 6-7 | `madewithml/data.py` | 5-6 |  |
| 1 | `import ray` | 8-11 | `diagnostics/eda.py` | 15-18 |  |
| 1 | `import ray` | 12-13 | `madewithml/data.py` | 8-9 |  |
| 1 | `import ray` | 14 | `madewithml/models.py` | 8 |  |
| 1 | `import ray` | 15-16 | `madewithml/data.py` | 11-12 |  |
| 1 | `import ray` | 17 | `madewithml/utils.py` | 3 |  |
| 1 | `import ray` | 18 | `madewithml/data.py` | 14 |  |
| 1 | `import ray` | 19 | `diagnostics/llm_baselines.py` | 23 |  |
| 1 | `import ray` | 20 | `madewithml/config.py` | 3 |  |
| 1 | `import ray` | 21 | `madewithml/utils.py` | 5 |  |
| 1 | `import ray` | 22-26 | `madewithml/config.py` | 5-9 |  |
| 2 | `if ray.is_initialized():` | 1-2 | `madewithml/train.py` | 156-157 | indented into train_model() |
| 3 | `DATASET_LOC = "https://raw.githubusercontent.com/GokuMohanda` | 1 | `madewithml/config.py` | 12 |  |
| 3 | `DATASET_LOC = "https://raw.githubusercontent.com/GokuMohanda` | 2-3 | `diagnostics/eda.py` | 26-27 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 4 | `df.shape` | 1 | `diagnostics/eda.py` | 30 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 5 | `df.columns` | 1 | `diagnostics/eda.py` | 33 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 6 | `df.tag.value_counts()` | 1 | `diagnostics/eda.py` | 36 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 7 | `train_df, val_df = train_test_split(df,stratify=df.tag,test_` | 1 | `diagnostics/eda.py` | 39 | indented into main() |
| 8 | `train_df.shape` | 1 | `diagnostics/eda.py` | 42 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 9 | `train_df.tag.value_counts()` | 1 | `diagnostics/eda.py` | 45 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 10 | `val_df.shape` | 1 | `diagnostics/eda.py` | 48 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 11 | `#verify if startify did its job` | 1-2 | `diagnostics/eda.py` | 51-52 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 12 | `val_df.shape` | 1 | `diagnostics/eda.py` | 55 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 13 | `all_tags=Counter(df['tag'])` | 1-5 | `diagnostics/eda.py` | 58-62 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 14 | `plt.figure(figsize=(10,5))` | 1-3 | `diagnostics/eda.py` | 65-67 | indented into main() |
| 15 | `# Plot tag frequencies #other alternative` | 1-9 | `diagnostics/eda.py` | 70-78 | indented into main() |
| 16 | `#most frequent tokens for each tag` | 1-9 | `diagnostics/eda.py` | 81-89 | indented into main() |
| 17 | `df['text']=df['title']+ " " + df['description']` | 1 | `diagnostics/eda.py` | 92 | indented into main() |
| 18 | `#lru cache can be used as well` | 1-17 | `madewithml/data.py` | 55-71 |  |
| 19 | `original_df=df.copy()` | 1-4 | `diagnostics/eda.py` | 95-98 | indented into main() |
| 20 | `df=df.drop(columns=['id','created_on','title','description']` | 1-4 | `diagnostics/eda.py` | 101-104 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 21 | `#label encoding` | 1-5 | `diagnostics/eda.py` | 107-111 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 22 | `df['tag']=df['tag'].map(class_to_index)` | 1-2 | `diagnostics/eda.py` | 114-115 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 23 | `def decode(indices,index_to_class):` | 1-3 | `madewithml/data.py` | 108-110 |  |
| 24 | `index_to_class={v:k for k,v in class_to_index.items()}` | 1-2 | `diagnostics/eda.py` | 118-119 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 25 | `#bert tokenizer test` | 1-9 | `diagnostics/eda.py` | 122-130 | indented into main() |
| 26 | `#method to tokenize` | 1-6 | `madewithml/data.py` | 74-79 |  |
| 27 | `tokenize(df.head(2))` | 1 | `diagnostics/eda.py` | 133 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 28 | `# here tokenize is the issue , it gets reloaded every batch ` | 1-10 | `madewithml/data.py` | 83-92 |  |
| 29 | `#just for refrence here. not used` | 1-23 | `diagnostics/ray_data_checks.py` | 20-42 |  |
| 30 | `preprocess(df=train_df,class_to_index=class_to_index)` | 1 | `diagnostics/eda.py` | 136 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 31 | `ray.data.DatasetContext.get_current().execution_options.pres` | 1-2 | `madewithml/train.py` | 159-160 | indented into train_model() |
| 32 | `ray.cluster_resources()` | 1 | `diagnostics/ray_data_checks.py` | 48 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 33 | `ds=ray.data.read_csv(DATASET_LOC,override_num_blocks=int(ray` | 1-3 | `diagnostics/ray_data_checks.py` | 51-53 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 34 | `#ray dataset` | 1-21 | `madewithml/data.py` | 31-51 |  |
| 35 | `train_ds,val_ds=stratify_split(ds,stratify='tag',test_size=T` | 1-2 | `diagnostics/ray_data_checks.py` | 56-57 | indented into main() |
| 36 | `#just to check  how to do in ray , earlier class to index ca` | 1-5 | `diagnostics/ray_data_checks.py` | 60-64 | indented into main() |
| 37 | `# sample_ds_classOne = train_ds.map_batches(` | 1-11 | `diagnostics/ray_data_checks.py` | 67-77 | indented into main() |
| 38 | `Lets establish baselines with zeroshot and few shot` | 1 | `diagnostics/llm_baselines.py` | 28 | markdown cell, copied as # comments |
| 39 | `client =anthropic.Anthropic()` | 1-2 | `diagnostics/llm_baselines.py` | 31-32 |  |
| 40 | `# response= client.messages.create(` | 1-6 | `diagnostics/llm_baselines.py` | 122-127 | indented into main() |
| 41 | `DATASET_LOC = "https://raw.githubusercontent.com/GokuMohanda` | 1-5 | `diagnostics/llm_baselines.py` | 130-134 | indented into main() |
| 42 | `HOLDOUT_LOC = "https://raw.githubusercontent.com/GokuMohanda` | 1 | `madewithml/config.py` | 14 |  |
| 42 | `HOLDOUT_LOC = "https://raw.githubusercontent.com/GokuMohanda` | 2 | `diagnostics/llm_baselines.py` | 137 | indented into main() |
| 43 | `test_df[['title','tag']][:3]` | 1 | `diagnostics/llm_baselines.py` | 140 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 44 | `# function that can predict tags for a given sample` | 1-20 | `diagnostics/llm_baselines.py` | 35-54 |  |
| 45 | `# List of dicts] as serving endpoint needs json format` | 1-3 | `diagnostics/llm_baselines.py` | 143-145 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 46 | `#sanity` | 1-16 | `diagnostics/llm_baselines.py` | 148-163 | indented into main() |
| 47 | `def get_predictions(inputs,model,system_content,assistant_co` | 1-14 | `diagnostics/llm_baselines.py` | 57-70 |  |
| 48 | `get_predictions(samples,model,system_context)` | 1 | `diagnostics/llm_baselines.py` | 166 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 49 | `def clean_predictions(y_pred,tags,default='other'):` | 1-8 | `diagnostics/llm_baselines.py` | 73-80 |  |
| 50 | `#plot ground truth and predictions` | 1-16 | `diagnostics/llm_baselines.py` | 83-98 |  |
| 51 | `def evaluate(test_df,model,system_content,tags,assistant_con` | 1-16 | `diagnostics/llm_baselines.py` | 101-116 |  |
| 52 | `#initialise dicts for analysis accros multiple models` | 1-3 | `diagnostics/llm_baselines.py` | 169-171 | indented into main() |
| 53 | `system_content = f"""` | 1-5 | `diagnostics/llm_baselines.py` | 174-178 | indented into main() |
| 54 | `method='zero_shot'` | 1-5 | `diagnostics/llm_baselines.py` | 181-185 | indented into main() |
| 55 | `#results seems to good , verifying with classification repor` | 1-3 | `diagnostics/llm_baselines.py` | 188-190 | indented into main() |
| 56 | `#zeroshot on haiku` | 1-6 | `diagnostics/llm_baselines.py` | 193-198 | indented into main() |
| 57 | `#pulling few shot examples from test set would be leakage` | 1-8 | `diagnostics/llm_baselines.py` | 201-208 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 58 | `# Add assistant context` | 1-3 | `diagnostics/llm_baselines.py` | 211-213 | indented into main() |
| 59 | `# Few-shot on sonnet 5` | 1-6 | `diagnostics/llm_baselines.py` | 216-221 | indented into main() |
| 60 | `# Few-shot on haiku` | 1-6 | `diagnostics/llm_baselines.py` | 224-229 | indented into main() |
| 61 | `print(classification_report(y_true=y_test,y_pred=y_pred[meth` | 1 | `diagnostics/llm_baselines.py` | 232 | indented into main() |
| 62 | `cm=confusion_matrix(y_test,y_pred[method][model],labels=tags` | 1-2 | `diagnostics/llm_baselines.py` | 235-236 | indented into main() |
| 63 | `print(json.dumps(performance,indent=2))` | 1 | `diagnostics/llm_baselines.py` | 239 | indented into main() |
| 64 | `by_model_context={}` | 1-7 | `diagnostics/llm_baselines.py` | 242-248 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 65 | `# Plotting the bar chart with metric scores on top of each b` | 1-23 | `diagnostics/llm_baselines.py` | 251-273 | indented into main() |
| 66 | `#another way as data is already in wider format , using pand` | 1-10 | `diagnostics/llm_baselines.py` | 276-285 | indented into main() |
| 67 | `#if was in longer formal then , example` | 1-17 | `diagnostics/llm_baselines.py` | 288-304 | indented into main() |
| 68 | `import random` | 1-2 | `madewithml/utils.py` | 7-8 |  |
| 68 | `import random` | 3 | `madewithml/data.py` | 16 |  |
| 69 | `def set_seeds(seed=42):` | 1-10 | `madewithml/utils.py` | 15-24 |  |
| 69 | `def set_seeds(seed=42):` | 11 | `madewithml/train.py` | 162 | indented into train_model() |
| 70 | `def load_data(num_samples=None):` | 1-6 | `madewithml/data.py` | 23-28 |  |
| 71 | `# The leading underscores (_fit, _transform_pandas) are the ` | 1-10 | `madewithml/data.py` | 95-104 |  |
| 72 | `import torch.nn as nn` | 1-2 | `madewithml/models.py` | 10-11 |  |
| 73 | `#PreTrained LLM` | 1-3 | `diagnostics/model_inspection.py` | 24-26 | indented into main() |
| 74 | `llm` | 1 | `diagnostics/model_inspection.py` | 29 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 75 | `print(list(llm.state_dict().keys()))` | 1-2 | `diagnostics/model_inspection.py` | 32-33 | indented into main() |
| 76 | `# for name , p in llm.named_parameters():` | 1-10 | `diagnostics/model_inspection.py` | 36-45 | indented into main() |
| 77 | `x=llm.named_parameters()` | 1-2 | `diagnostics/model_inspection.py` | 48-49 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 78 | `y=list(llm.parameters())` | 1-2 | `diagnostics/model_inspection.py` | 52-53 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 79 | `llm.state_dict()` | 1-2 | `diagnostics/model_inspection.py` | 56-57 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 80 | `# Sample sanity` | 1-6 | `diagnostics/model_inspection.py` | 60-65 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 81 | `import torch.nn.functional as F` | 1 | `madewithml/models.py` | 13 |  |
| 82 | `class FinetunedLLM(nn.Module):` | 1-50 | `madewithml/models.py` | 17-66 |  |
| 83 | `#intitialize model` | 1-4 | `diagnostics/model_inspection.py` | 68-71 | indented into main() |
| 84 | `llm.encoder.layer[1].attention.self.key.weight` | 1 | `diagnostics/model_inspection.py` | 74 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 85 | `from ray.train.torch import get_device as ray_get_device` | 1-2 | `madewithml/utils.py` | 10-11 |  |
| 85 | `from ray.train.torch import get_device as ray_get_device` | 3-10 | `madewithml/utils.py` | 27-34 |  |
| 86 | `def pad_array(arr, dtype=np.int32):` | 1-6 | `madewithml/utils.py` | 38-43 |  |
| 87 | `def collate_fn(batch):` | 1-5 | `madewithml/utils.py` | 47-51 |  |
| 88 | `# Sample batch\| dataset > shard (1 per worker) > block > ro` | 1-3 | `diagnostics/ray_data_checks.py` | 80-82 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 89 | `# Ray 2.58: ray.air.Checkpoint / ray.air.session were remove` | 1-12 | `madewithml/train.py` | 10-21 |  |
| 90 | `def train_step(ds, batch_size, model, num_classes, loss_fn, ` | 1-14 | `madewithml/train.py` | 34-47 |  |
| 91 | `def eval_step(ds, batch_size, model, num_classes, loss_fn):` | 1-14 | `madewithml/train.py` | 51-64 |  |
| 92 | `trainer.fit()` | 1-21 | `madewithml/train.py` | 68-88 | markdown cell, copied as # comments |
| 93 | `# Training loop` | 1-44 | `madewithml/train.py` | 91-134 |  |
| 94 | `#if imbalanced` | 1-11 | `madewithml/train.py` | 136-146 |  |
| 95 | `# Train loop config` | 1-2 | `madewithml/train.py` | 177-178 | indented into train_model() |
| 95 | `# Train loop config` | 3-8 | `madewithml/config.py` | 19-24 |  |
| 95 | `# Train loop config` | 9-10 | `madewithml/train.py` | 181-182 | indented into train_model() |
| 96 | `# Workers -- this machine is CPU-only (torch.cuda.is_availab` | 1-12 | `madewithml/train.py` | 186-197 | indented into train_model() |
| 97 | `# Scaling config` | 1-7 | `madewithml/train.py` | 202-208 | indented into train_model() |
| 98 | `# Run config` | 1-8 | `madewithml/train.py` | 210-217 | indented into train_model() |
| 99 | `ds = load_data()` | 1 | `madewithml/train.py` | 164 | MODIFIED: passes num_samples (None = full dataset, as in the notebook) |
| 99 | `ds = load_data()` | 2 | `madewithml/train.py` | 165 | indented into train_model() |
| 100 | `# Preprocess` | 1-6 | `madewithml/train.py` | 168-173 | indented into train_model() |
| 101 | `# Dataset config` | 1 | `madewithml/train.py` | 219 | indented into train_model() |
| 101 | `# Dataset config` | 2 | `madewithml/train.py` | 23 |  |
| 101 | `# Dataset config` | 3-8 | `madewithml/train.py` | 221-226 | indented into train_model() |
| 102 | `# The driver still holds two SciBERT copies from the explora` | 1-2 | `diagnostics/model_inspection.py` | 77-78 | indented into main() |
| 102 | `# The driver still holds two SciBERT copies from the explora` | 3 | `madewithml/train.py` | 25 |  |
| 102 | `# The driver still holds two SciBERT copies from the explora` | 4 | `diagnostics/model_inspection.py` | 80 | indented into main() |
| 102 | `# The driver still holds two SciBERT copies from the explora` | 5-17 | `madewithml/train.py` | 228-240 | indented into train_model() |
| 103 | `results = trainer.fit()` | 1 | `madewithml/train.py` | 243 | indented into train_model() |
| 104 | `results.metrics_dataframe` | 1 | `madewithml/train.py` | 245 | MODIFIED: wrapped in print() (a bare expression shows nothing in a script) |
| 105 | `results.best_checkpoints` | 1 | `madewithml/train.py` | 247 | MODIFIED: wrapped in print() (a bare expression shows nothing in a script) |
| 106 | (empty cell) | – | – | – | nothing to copy |

## By file

### `madewithml/__init__.py`

| file lines | from nb cell | cell lines | note |
|---|---:|---|---|
| – | – | – | only added lines |

### `madewithml/config.py`

| file lines | from nb cell | cell lines | note |
|---|---:|---|---|
| 3 | 1 | 20 |  |
| 5-9 | 1 | 22-26 |  |
| 12 | 3 | 1 |  |
| 14 | 42 | 1 |  |
| 19-24 | 95 | 3-8 |  |

### `madewithml/utils.py`

| file lines | from nb cell | cell lines | note |
|---|---:|---|---|
| 3 | 1 | 17 |  |
| 5 | 1 | 21 |  |
| 7-8 | 68 | 1-2 |  |
| 10-11 | 85 | 1-2 |  |
| 15-24 | 69 | 1-10 |  |
| 27-34 | 85 | 3-10 |  |
| 38-43 | 86 | 1-6 |  |
| 47-51 | 87 | 1-5 |  |

### `madewithml/data.py`

| file lines | from nb cell | cell lines | note |
|---|---:|---|---|
| 3 | 1 | 1 |  |
| 5-6 | 1 | 6-7 |  |
| 8-9 | 1 | 12-13 |  |
| 11-12 | 1 | 15-16 |  |
| 14 | 1 | 18 |  |
| 16 | 68 | 3 |  |
| 23-28 | 70 | 1-6 |  |
| 31-51 | 34 | 1-21 |  |
| 55-71 | 18 | 1-17 |  |
| 74-79 | 26 | 1-6 |  |
| 83-92 | 28 | 1-10 |  |
| 95-104 | 71 | 1-10 |  |
| 108-110 | 23 | 1-3 |  |

### `madewithml/models.py`

| file lines | from nb cell | cell lines | note |
|---|---:|---|---|
| 6 | 1 | 5 |  |
| 8 | 1 | 14 |  |
| 10-11 | 72 | 1-2 |  |
| 13 | 81 | 1 |  |
| 17-66 | 82 | 1-50 |  |

### `madewithml/train.py`

| file lines | from nb cell | cell lines | note |
|---|---:|---|---|
| 10-21 | 89 | 1-12 |  |
| 23 | 101 | 2 |  |
| 25 | 102 | 3 |  |
| 34-47 | 90 | 1-14 |  |
| 51-64 | 91 | 1-14 |  |
| 68-88 | 92 | 1-21 | markdown cell, copied as # comments |
| 91-134 | 93 | 1-44 |  |
| 136-146 | 94 | 1-11 |  |
| 156-157 | 2 | 1-2 | indented into train_model() |
| 159-160 | 31 | 1-2 | indented into train_model() |
| 162 | 69 | 11 | indented into train_model() |
| 164 | 99 | 1 | MODIFIED: passes num_samples (None = full dataset, as in the notebook) |
| 165 | 99 | 2 | indented into train_model() |
| 168-173 | 100 | 1-6 | indented into train_model() |
| 177-178 | 95 | 1-2 | indented into train_model() |
| 181-182 | 95 | 9-10 | indented into train_model() |
| 186-197 | 96 | 1-12 | indented into train_model() |
| 202-208 | 97 | 1-7 | indented into train_model() |
| 210-217 | 98 | 1-8 | indented into train_model() |
| 219 | 101 | 1 | indented into train_model() |
| 221-226 | 101 | 3-8 | indented into train_model() |
| 228-240 | 102 | 5-17 | indented into train_model() |
| 243 | 103 | 1 | indented into train_model() |
| 245 | 104 | 1 | MODIFIED: wrapped in print() (a bare expression shows nothing in a script) |
| 247 | 105 | 1 | MODIFIED: wrapped in print() (a bare expression shows nothing in a script) |

### `diagnostics/eda.py`

| file lines | from nb cell | cell lines | note |
|---|---:|---|---|
| 15-18 | 1 | 8-11 |  |
| 26-27 | 3 | 2-3 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 30 | 4 | 1 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 33 | 5 | 1 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 36 | 6 | 1 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 39 | 7 | 1 | indented into main() |
| 42 | 8 | 1 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 45 | 9 | 1 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 48 | 10 | 1 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 51-52 | 11 | 1-2 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 55 | 12 | 1 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 58-62 | 13 | 1-5 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 65-67 | 14 | 1-3 | indented into main() |
| 70-78 | 15 | 1-9 | indented into main() |
| 81-89 | 16 | 1-9 | indented into main() |
| 92 | 17 | 1 | indented into main() |
| 95-98 | 19 | 1-4 | indented into main() |
| 101-104 | 20 | 1-4 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 107-111 | 21 | 1-5 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 114-115 | 22 | 1-2 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 118-119 | 24 | 1-2 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 122-130 | 25 | 1-9 | indented into main() |
| 133 | 27 | 1 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 136 | 30 | 1 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |

### `diagnostics/ray_data_checks.py`

| file lines | from nb cell | cell lines | note |
|---|---:|---|---|
| 20-42 | 29 | 1-23 |  |
| 48 | 32 | 1 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 51-53 | 33 | 1-3 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 56-57 | 35 | 1-2 | indented into main() |
| 60-64 | 36 | 1-5 | indented into main() |
| 67-77 | 37 | 1-11 | indented into main() |
| 80-82 | 88 | 1-3 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |

### `diagnostics/model_inspection.py`

| file lines | from nb cell | cell lines | note |
|---|---:|---|---|
| 24-26 | 73 | 1-3 | indented into main() |
| 29 | 74 | 1 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 32-33 | 75 | 1-2 | indented into main() |
| 36-45 | 76 | 1-10 | indented into main() |
| 48-49 | 77 | 1-2 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 52-53 | 78 | 1-2 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 56-57 | 79 | 1-2 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 60-65 | 80 | 1-6 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 68-71 | 83 | 1-4 | indented into main() |
| 74 | 84 | 1 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 77-78 | 102 | 1-2 | indented into main() |
| 80 | 102 | 4 | indented into main() |

### `diagnostics/llm_baselines.py`

| file lines | from nb cell | cell lines | note |
|---|---:|---|---|
| 19-21 | 1 | 2-4 |  |
| 23 | 1 | 19 |  |
| 28 | 38 | 1 | markdown cell, copied as # comments |
| 31-32 | 39 | 1-2 |  |
| 35-54 | 44 | 1-20 |  |
| 57-70 | 47 | 1-14 |  |
| 73-80 | 49 | 1-8 |  |
| 83-98 | 50 | 1-16 |  |
| 101-116 | 51 | 1-16 |  |
| 122-127 | 40 | 1-6 | indented into main() |
| 130-134 | 41 | 1-5 | indented into main() |
| 137 | 42 | 2 | indented into main() |
| 140 | 43 | 1 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 143-145 | 45 | 1-3 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 148-163 | 46 | 1-16 | indented into main() |
| 166 | 48 | 1 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 169-171 | 52 | 1-3 | indented into main() |
| 174-178 | 53 | 1-5 | indented into main() |
| 181-185 | 54 | 1-5 | indented into main() |
| 188-190 | 55 | 1-3 | indented into main() |
| 193-198 | 56 | 1-6 | indented into main() |
| 201-208 | 57 | 1-8 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 211-213 | 58 | 1-3 | indented into main() |
| 216-221 | 59 | 1-6 | indented into main() |
| 224-229 | 60 | 1-6 | indented into main() |
| 232 | 61 | 1 | indented into main() |
| 235-236 | 62 | 1-2 | indented into main() |
| 239 | 63 | 1 | indented into main() |
| 242-248 | 64 | 1-7 | indented into main(); 1 line(s) wrapped in print(), see Modified lines |
| 251-273 | 65 | 1-23 | indented into main() |
| 276-285 | 66 | 1-10 | indented into main() |
| 288-304 | 67 | 1-17 | indented into main() |

## Modified lines

| nb cell | cell line | notebook | file | file line | now | why |
|---:|---:|---|---|---:|---|---|
| 99 | 1 | `ds = load_data()` | `madewithml/train.py` | 164 | `ds = load_data(num_samples=num_samples)` | passes num_samples (None = full dataset, as in the notebook) |
| 104 | 1 | `results.metrics_dataframe` | `madewithml/train.py` | 245 | `print(results.metrics_dataframe)` | wrapped in print() (a bare expression shows nothing in a script) |
| 105 | 1 | `results.best_checkpoints` | `madewithml/train.py` | 247 | `print(results.best_checkpoints)` | wrapped in print() (a bare expression shows nothing in a script) |
| 3 | 3 | `df.head()` | `diagnostics/eda.py` | 27 | `print(df.head())` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 4 | 1 | `df.shape` | `diagnostics/eda.py` | 30 | `print(df.shape)` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 5 | 1 | `df.columns` | `diagnostics/eda.py` | 33 | `print(df.columns)` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 6 | 1 | `df.tag.value_counts()` | `diagnostics/eda.py` | 36 | `print(df.tag.value_counts())` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 8 | 1 | `train_df.shape` | `diagnostics/eda.py` | 42 | `print(train_df.shape)` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 9 | 1 | `train_df.tag.value_counts()` | `diagnostics/eda.py` | 45 | `print(train_df.tag.value_counts())` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 10 | 1 | `val_df.shape` | `diagnostics/eda.py` | 48 | `print(val_df.shape)` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 11 | 2 | `val_df.tag.value_counts() * int((1-TEST_SIZE)/TEST_SIZE)` | `diagnostics/eda.py` | 52 | `print(val_df.tag.value_counts() * int((1-TEST_SIZE)/TEST_SIZE))` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 12 | 1 | `val_df.shape` | `diagnostics/eda.py` | 55 | `print(val_df.shape)` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 13 | 2 | `all_tags.most_common()` | `diagnostics/eda.py` | 59 | `print(all_tags.most_common())` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 20 | 4 | `df.head()` | `diagnostics/eda.py` | 104 | `print(df.head())` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 21 | 5 | `class_to_index` | `diagnostics/eda.py` | 111 | `print(class_to_index)` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 22 | 2 | `df.head()` | `diagnostics/eda.py` | 115 | `print(df.head())` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 24 | 2 | `decode(df.head(1)['tag'].values,index_to_class=index_to_class)` | `diagnostics/eda.py` | 119 | `print(decode(df.head(1)['tag'].values,index_to_class=index_to_class))` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 27 | 1 | `tokenize(df.head(2))` | `diagnostics/eda.py` | 133 | `print(tokenize(df.head(2)))` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 30 | 1 | `preprocess(df=train_df,class_to_index=class_to_index)` | `diagnostics/eda.py` | 136 | `print(preprocess(df=train_df,class_to_index=class_to_index))` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 32 | 1 | `ray.cluster_resources()` | `diagnostics/ray_data_checks.py` | 48 | `print(ray.cluster_resources())` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 33 | 3 | `ds.take(1)` | `diagnostics/ray_data_checks.py` | 53 | `print(ds.take(1))` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 88 | 3 | `collate_fn(batch=sample_batch)` | `diagnostics/ray_data_checks.py` | 82 | `print(collate_fn(batch=sample_batch))` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 74 | 1 | `llm` | `diagnostics/model_inspection.py` | 29 | `print(llm)` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 77 | 2 | `list(x)` | `diagnostics/model_inspection.py` | 49 | `print(list(x))` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 78 | 2 | `y` | `diagnostics/model_inspection.py` | 53 | `print(y)` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 79 | 1 | `llm.state_dict()` | `diagnostics/model_inspection.py` | 56 | `print(llm.state_dict())` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 80 | 6 | `np.shape(seq), np.shape(pool)` | `diagnostics/model_inspection.py` | 65 | `print(np.shape(seq), np.shape(pool))` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 84 | 1 | `llm.encoder.layer[1].attention.self.key.weight` | `diagnostics/model_inspection.py` | 74 | `print(llm.encoder.layer[1].attention.self.key.weight)` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 43 | 1 | `test_df[['title','tag']][:3]` | `diagnostics/llm_baselines.py` | 140 | `print(test_df[['title','tag']][:3])` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 45 | 3 | `samples` | `diagnostics/llm_baselines.py` | 145 | `print(samples)` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 48 | 1 | `get_predictions(samples,model,system_context)` | `diagnostics/llm_baselines.py` | 166 | `print(get_predictions(samples,model,system_context))` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 57 | 8 | `additional_context` | `diagnostics/llm_baselines.py` | 208 | `print(additional_context)` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |
| 64 | 7 | `by_model_context` | `diagnostics/llm_baselines.py` | 248 | `print(by_model_context)` | wrapped in print() (the notebook displayed this value; a script only shows what it prints) |

## Added lines (not in the notebook)

Imports each file needs, the `train_model()` and `main()` wrappers, and a few lines the scripts need. Header comments (`# [nb cell N, ...]`) are omitted here.

| file | line | added |
|---|---:|---|
| `madewithml/__init__.py` | 1 | `"""madewithml: SciBERT tag classifier, refactored from test.ipynb."""` |
| `madewithml/config.py` | 1 | `"""Constants, training hyperparameters and logging setup (from notebook cells 1, 3, 42, 95)."""` |
| `madewithml/config.py` | 16 | `# Training hyperparameters used by madewithml/train.py (num_classes is added there, from the data)` |
| `madewithml/config.py` | 17 | `TRAIN_LOOP_CONFIG = {` |
| `madewithml/config.py` | 25 | `}` |
| `madewithml/utils.py` | 1 | `"""Seeds, device lookup, padding and the collate function."""` |
| `madewithml/data.py` | 1 | `"""Loading, splitting and preprocessing the dataset."""` |
| `madewithml/data.py` | 17 | `import numpy as np` |
| `madewithml/data.py` | 19 | `from madewithml.config import DATASET_LOC, RANDOM_SEED` |
| `madewithml/models.py` | 1 | `"""FinetunedLLM: SciBERT encoder + classification head."""` |
| `madewithml/models.py` | 2 | `import os` |
| `madewithml/models.py` | 4 | `import torch` |
| `madewithml/train.py` | 1 | `"""Distributed training with Ray Train (train_step, eval_step, train_loop_per_worker, train_model)."""` |
| `madewithml/train.py` | 2 | `import os` |
| `madewithml/train.py` | 4 | `import numpy as np` |
| `madewithml/train.py` | 5 | `import ray` |
| `madewithml/train.py` | 6 | `import torch` |
| `madewithml/train.py` | 7 | `import torch.nn as nn` |
| `madewithml/train.py` | 8 | `from transformers import BertModel` |
| `madewithml/train.py` | 27 | `from madewithml.config import TEST_SIZE, TRAIN_LOOP_CONFIG` |
| `madewithml/train.py` | 28 | `from madewithml.data import CustomPreprocessor, load_data, stratify_split` |
| `madewithml/train.py` | 29 | `from madewithml.models import FinetunedLLM` |
| `madewithml/train.py` | 30 | `from madewithml.utils import collate_fn, set_seeds` |
| `madewithml/train.py` | 149 | `def train_model(num_samples=None, num_workers_override=None, **config_overrides):` |
| `madewithml/train.py` | 150 | `"""Notebook cells 2, 31, 69 (last line) and 95-105 as one function.` |
| `madewithml/train.py` | 152 | `Defaults reproduce the notebook run. For a quick check:` |
| `madewithml/train.py` | 153 | `train_model(num_samples=100, num_workers_override=1, num_epochs=1)` |
| `madewithml/train.py` | 154 | `"""` |
| `madewithml/train.py` | 174 | `num_classes = len(preprocessor.class_to_index)  # [added] the notebook took this from EDA cell 21` |
| `madewithml/train.py` | 179 | `**TRAIN_LOOP_CONFIG,  # [added] the hyperparameter values now live in madewithml/config.py` |
| `madewithml/train.py` | 183 | `train_loop_config.update(config_overrides)  # [added] e.g. num_epochs=1 for a quick run` |
| `madewithml/train.py` | 198 | `if num_workers_override:  # [added] e.g. 1 worker for a quick run` |
| `madewithml/train.py` | 199 | `num_workers = num_workers_override` |
| `madewithml/train.py` | 248 | `return results` |
| `madewithml/train.py` | 251 | `if __name__ == "__main__":` |
| `madewithml/train.py` | 252 | `train_model()` |
| `diagnostics/eda.py` | 1 | `"""EDA on the raw CSV with pandas: shapes, tag balance, plots, word clouds, text cleaning, tokenizer checks.` |
| `diagnostics/eda.py` | 3 | `Run from the project root:` |
| `diagnostics/eda.py` | 4 | `python diagnostics/eda.py` |
| `diagnostics/eda.py` | 5 | `"""` |
| `diagnostics/eda.py` | 6 | `import sys` |
| `diagnostics/eda.py` | 7 | `from pathlib import Path` |
| `diagnostics/eda.py` | 9 | `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets 'import madewithml' work from any folder` |
| `diagnostics/eda.py` | 11 | `import pandas as pd` |
| `diagnostics/eda.py` | 12 | `from sklearn.model_selection import train_test_split` |
| `diagnostics/eda.py` | 13 | `from transformers import BertTokenizer` |
| `diagnostics/eda.py` | 20 | `from madewithml.config import DATASET_LOC, RANDOM_SEED, TEST_SIZE` |
| `diagnostics/eda.py` | 21 | `from madewithml.data import clean_text, decode, preprocess, tokenize` |
| `diagnostics/eda.py` | 24 | `def main():` |
| `diagnostics/eda.py` | 138 | `plt.show()  # [added] display figures created after the last plt.show() (the word clouds)` |
| `diagnostics/eda.py` | 141 | `if __name__ == "__main__":` |
| `diagnostics/eda.py` | 142 | `main()` |
| `diagnostics/ray_data_checks.py` | 1 | `"""Ray Data checks: cluster resources, stratified split, class mapping, preprocessing, one collated batch.` |
| `diagnostics/ray_data_checks.py` | 3 | `Run from the project root:` |
| `diagnostics/ray_data_checks.py` | 4 | `python diagnostics/ray_data_checks.py` |
| `diagnostics/ray_data_checks.py` | 5 | `"""` |
| `diagnostics/ray_data_checks.py` | 6 | `import sys` |
| `diagnostics/ray_data_checks.py` | 7 | `from pathlib import Path` |
| `diagnostics/ray_data_checks.py` | 9 | `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets 'import madewithml' work from any folder` |
| `diagnostics/ray_data_checks.py` | 11 | `import numpy as np` |
| `diagnostics/ray_data_checks.py` | 12 | `import ray` |
| `diagnostics/ray_data_checks.py` | 13 | `from transformers import BertTokenizer` |
| `diagnostics/ray_data_checks.py` | 15 | `from madewithml.config import DATASET_LOC, RANDOM_SEED, TEST_SIZE` |
| `diagnostics/ray_data_checks.py` | 16 | `from madewithml.data import clean_text, preprocess, stratify_split` |
| `diagnostics/ray_data_checks.py` | 17 | `from madewithml.utils import collate_fn` |
| `diagnostics/ray_data_checks.py` | 44 | `def main():` |
| `diagnostics/ray_data_checks.py` | 45 | `ray.init(ignore_reinit_error=True)  # [added] train.py owns the notebook's ray.init() (cell 31)` |
| `diagnostics/ray_data_checks.py` | 85 | `if __name__ == "__main__":` |
| `diagnostics/ray_data_checks.py` | 86 | `main()` |
| `diagnostics/model_inspection.py` | 1 | `"""Inspect SciBERT: parameters, buffers, state_dict, a forward pass, the FinetunedLLM head.` |
| `diagnostics/model_inspection.py` | 3 | `Run from the project root:` |
| `diagnostics/model_inspection.py` | 4 | `python diagnostics/model_inspection.py` |
| `diagnostics/model_inspection.py` | 5 | `"""` |
| `diagnostics/model_inspection.py` | 6 | `import sys` |
| `diagnostics/model_inspection.py` | 7 | `from pathlib import Path` |
| `diagnostics/model_inspection.py` | 9 | `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets 'import madewithml' work from any folder` |
| `diagnostics/model_inspection.py` | 11 | `import numpy as np` |
| `diagnostics/model_inspection.py` | 12 | `import pandas as pd` |
| `diagnostics/model_inspection.py` | 13 | `from transformers import BertModel, BertTokenizer` |
| `diagnostics/model_inspection.py` | 15 | `from madewithml.config import DATASET_LOC` |
| `diagnostics/model_inspection.py` | 16 | `from madewithml.models import FinetunedLLM` |
| `diagnostics/model_inspection.py` | 19 | `def main():` |
| `diagnostics/model_inspection.py` | 20 | `tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")  # [added] notebook defined it i` |
| `diagnostics/model_inspection.py` | 21 | `num_classes = pd.read_csv(DATASET_LOC).tag.nunique()  # [added] notebook defined it in EDA cell 21` |
| `diagnostics/model_inspection.py` | 83 | `if __name__ == "__main__":` |
| `diagnostics/model_inspection.py` | 84 | `main()` |
| `diagnostics/llm_baselines.py` | 1 | `"""Zero-/few-shot baselines with Claude on the holdout set.` |
| `diagnostics/llm_baselines.py` | 3 | `Calls the Anthropic API: needs ANTHROPIC_API_KEY and costs tokens.` |
| `diagnostics/llm_baselines.py` | 4 | `Run from the project root:` |
| `diagnostics/llm_baselines.py` | 5 | `python diagnostics/llm_baselines.py` |
| `diagnostics/llm_baselines.py` | 6 | `"""` |
| `diagnostics/llm_baselines.py` | 7 | `import sys` |
| `diagnostics/llm_baselines.py` | 8 | `from pathlib import Path` |
| `diagnostics/llm_baselines.py` | 10 | `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets 'import madewithml' work from any folder` |
| `diagnostics/llm_baselines.py` | 12 | `import json` |
| `diagnostics/llm_baselines.py` | 13 | `from collections import Counter` |
| `diagnostics/llm_baselines.py` | 15 | `import matplotlib.pyplot as plt` |
| `diagnostics/llm_baselines.py` | 16 | `import pandas as pd` |
| `diagnostics/llm_baselines.py` | 17 | `import seaborn as sns` |
| `diagnostics/llm_baselines.py` | 25 | `from madewithml.config import HOLDOUT_LOC` |
| `diagnostics/llm_baselines.py` | 119 | `def main():` |
| `diagnostics/llm_baselines.py` | 307 | `if __name__ == "__main__":` |
| `diagnostics/llm_baselines.py` | 308 | `main()` |
