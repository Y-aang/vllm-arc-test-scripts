# Test Scripts for vLLM-ARC

## Document QA Test (Example: QuALITY, WikiQA)
### Step 0: Prepare Environment
Install [vLLM-ARC](https://github.com/Y-aang/vllm-ARC). Build from source (Reference to vLLM official manual).
### Step 1: Download Dataset
For QuALITY, the dataset will be downloaded automically in step 2.

For WikiQA, download and process data using [`download_wikiqa.py`](synthesis/WikiQA/download_wikiqa.py).
```
python download_wikiqa.py
```
### Step 2: Sample from Dataset
```
# Sample from QuALITY
python ./synthesis/Quality/quality_distshift_sample.py
# Sample from WikiQA
python ./synthesis/WikiQA/wikiqa_distshift_sample.py
```
### Step 3: Run Experiment
```
# Modify setting (model, device, datapath, runtime settings) in ./synthesis/model_config.py, test_script_batch.py (# Step 1: Parameters)
# Set experiment parameters (cache size, cache strategy) in ./synthesis/quality_dist_test.sh, wikiqa_dist_test.sh
# Experiment logs would be generated under ./synthesis/
bash ./synthesis/quality_dist_test.sh
bash ./synthesis/wikiqa_dist_test.sh
```
### Step 4: Collect Data
```
# Modify information for experiment you want to collect in ./synthesis/collect_result.py
python ./synthesis/collect_result.py
```
## Conversation Test (Example: Qwen-Bailian)
### Step 0: Prepare Environment
Install [vLLM-ARC](https://github.com/Y-aang/vllm-ARC). Build from source (Reference to vLLM official manual).
### Step 1: Download Dataset
Download [Trace A](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon) `qwen_traceA_blksz_16.jsonl`.
### Step 2: Process Dataset
```
# Process data
python ./burst/qwen-bailian-usagetraces-anon/process_json.py
python ./burst/qwen-bailian-usagetraces-anon/process_batch_inference.py
```
### Step 3: Run Experiment
```
# Modify setting (model, device, datapath, runtime settings) in ./synthesis/model_config.py, test_script_batch (# Step 1: Parameters)
# Set experiment parameters (cache size, cache strategy) in ./synthesis/wild_batch.sh
# Experiment logs would be generated under ./synthesis/
bash ./synthesis/wild_batch.sh
```
### Step 4: Collect Data
```
# Modify information for experiment you want to collect in ./synthesis/collect_result.py
python ./synthesis/collect_result.py
```

## HotSpot Sampling Verification (Example: SQuAD)
### Step 0: Prepare Environment
Install [vLLM-ARC](https://github.com/Y-aang/vllm-ARC). Build from source (Reference to vLLM official manual).
### Step 1: HotSpot sampling
```
python squad_hotspot_sample.py > ./sample/squad_hotspot_sample.txt 2>&1
```
### Step 2: Run Experiment
Manually set cache strategy in `Qwen2.5_script.sh` and vLLM's `make_evictor()`. 

Fill in the configuration in `model_config.py` as instructed in the code comments.
```
python squad_hotspot_sample.py > ./sample/squad_hotspot_sample.txt 2>&1
```
### Step 3: Collect Data
```
python collect_result.py
```
### Step 4: Plot the charts
Copy the `.csv`. Use `view_graph.ipynb`. (From [Simulator tool box](https://github.com/Y-aang/vLLM-Eviction-Simulator.git)) 
## Some Useful Scripts
- `cache_evicion.py`: Set up several sequences to fill the cache and observe the evictor's behavior under hit and miss conditions.
- `blocksize_batch_exp.sh`, `blocksize_batch_exp.sh`: for block size experiment.
- `positional_dependency.sh`, `positional_dependency.py`: for positional dependency experiment.
- `wikiQA_2Q_valid.py`, `wikiQA_2Q_valid_cut.py`
- `./synthesis/`: for HotSpot and Distribution Shift experiments.
- `./data_process/`: process wikiQA and view results.
