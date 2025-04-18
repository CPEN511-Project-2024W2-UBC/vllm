# CPEN 511 Project:

Presented by:
- Tom Wang
- Jiajun Huang

from the University of British Columbia

## Introduction

This work is based on the implementatino of vllm open source project. The [original readme](Original_README.md) can be found in the same directory. The offical website of vllm is [here](https://vllm.ai/).

Build the vLLM using the 
```bash
pip install -e .
```
## Getting Started

To install vllm, you can use the following command:

```bash
git clone git@github.com:CPEN511-Project-2024W2-UBC/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .
```
for NVIDIA CUDA users. For other options, check the [official installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source).However, no test has been done on non-NVIDIA CUDA environment for our project.

## Running the code

To run a simple test to demenstrate our work, you can use the following command:

```bash
cd playground
python test.py
```
`test.py` will run a [Naive Search](#naive-search) of a given LLM model. By default, it will use `Facebook/opt-125m` model. You can change the model in the script following the examples in the script. A few log files will be generated in the `playground` directory. The log files contain the following information for each search:
- number of gpu and cpu blocks
- number of swap operations
- number of blocks swapped out
- output throughput and input traffic

To visualize the results, you can use the following command:
```bash
python visualize.py
```

This will generate 4 plots in the `playground` directory. The plots are:
- Search factor vs. output throughput
- Search factor vs. input traffic
- Search factor vs. number of swap operations
- Search factor vs. number of blocks swapped out

### Running Naive Search 2

For [Naive Search 2](#running-naive-search-2), there must be some code modification, thus we need to:
``` bash
git checkout Naive2
```
This will switch the code to the branch that contains the implementation of Naive Search 2. Then you can run the same command as above to run the test. The only difference is that now it searches for the maximum average space for the sequences in the KV cache. The search space is $[1.8, 7.5]$ and the step size is $0.2$.

To see the default search on FB/opt-125m, you can run the following command:
```bash
cd playground # if you are not in the directory
python aim_for_av_7.py
```
This will run the test with the default search space. The log files will be generated in the `playground` directory. The log files contain the same information as above.
The only difference is that now it searches for the maximum average space for the sequences in the KV cache. The search space is $[1.8, 7.5]$ and the step size is $0.2$.

To visualize the results, you can use the following command:
```bash
python visualize.py
```
This will generate 4 plots in the `playground` directory same as previous section.

### To visualize all the results
`vllm/playgruond/plot_multiple.ipynb` contains the code to put all results in one plot. You can run the code in the notebook to generate the plots. Make sure the change `places_to_chack{}` to the directory where you put the log files. 

### Gather Memory Operation traces
By default, no trace will be collected. To collect the memory operation traces, you need to change `vllm/core/logger.py`

Instead of 
```python
logger.setLevel(logging.CRITICAL)
# logger.setLevel(logging.DEBUG)
```
You need to change it to 
```python
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.CRITICAL)
```

With this, after running one workload, memory operation trace will appear in `debug.log` file.

## Naive Search

Let $n$ be number of sequences schduled in KV cache. The naive implementation of the KV cache is to assume that the probability of a sequence that needs to be appended to be $\frac{1}{16}$, then we do a search of how many blocks each sequence will need to append. That is, let $o\in\mathbb{R}$, we test $o$ till we find an optimal solution. 

For pratical reason, we test $o \in [0,12]$ with step size $0.2$. The result can be obtained from [running the code](#running-the-code). 

## Naive Search 2

For the second search, we make the hypothesis that the we aviod swapping by controlling the number of sequences in the KV cache. 

Let $N$ be the number of sequences in the KV cache, $b$ be the number of blocks in the KV cache. The average number of blocks in the KV cache is $b/N$. From observation, we see that average sequence length in these workloads are about 7.5 and average space needed for prompts are 2. Thus the search space is $[1.8, 7.5]$. The reset of the setting the same as [Naive Search 1](#naive-search)

## Run the Sequence prediction 
Go to the file of `vllm/cpen511/new_memery_predict/train_reg.py` and `vllm/cpen511/new_memery_predict/train_cls.py`, change the `file_path` variable to the data file which is at `vllm/cpen511/pure_sequence.csv` 
You can change the `windows_size` and `mod` to check the accuracy.

To run the classcification model:
```bash
python vllm/cpen511/new_memery_predict/train_cls.py
```

To run the regression model:
```bash
python vllm/cpen511/new_memery_predict/train_reg.py
```

