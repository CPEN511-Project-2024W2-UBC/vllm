# CPEN 511 Project:

Presented by:
- Tom Wang
- Jiajun Huang

from the University of British Columbia

## Introduction

This work is based on the implementation of vllm open source project. The [original readme](Original_README.md) can be found in the same directory. The offical website of vllm is [here](https://vllm.ai/).

## Getting Started

To install vllm, you can use the following command:

```bash
git clone git@github.com:CPEN511-Project-2024W2-UBC/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .
```
for NVIDIA CUDA users. For other options, check the [official installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source). However, no test has been done on non-NVIDIA CUDA environment for our project.

## Running the code

To run a simple test to demenstrate our work, you can use the following command:

```bash
cd playground
git clone https://github.com/centerforaisafety/HarmBench.git
python test.py
```
This will run a simple test that demonstrates an [Naive Search](#naive-implementation) of a default set up of [HarmBench Dataset](#harmbench-dataset)models and KV cache. The test will run a few iterations and print out the results.

To visualize the results, you can use the following command:

```bash
python visualize.py
```
This will generate a plot of the results of the test. The plot will show the number of sequences that need to be appended to the KV cache over time.
The plot will be saved in the `playground` directory as `*.png`.

## HarmBench Dataset

HarmBench Dataset is available on [Github](https://github.com/centerforaisafety/HarmBench). We just picked some LLM input promts as our test cases. The dataset is designed to benchmark the performance of LLMs in terms of their ability to generate harmful content. It includes a variety of prompts that are designed to elicit harmful responses from LLMs. However, for this project, we only use the prompts as our test cases.

## Naive Implementation

Let $n$ be number of sequences schduled in KV cache. The naive implementation of the KV cache is to assume that the probability of a sequence that needs to be appended to be $\frac{1}{16}$, then we do a search of how many blocks each sequence will need to append. That is, let $o\in\mathbb{R}$, we test $o$ till we find an optimal solution. 

For pratical reason, we test $o \in [0,12]$ with step size $0.2$. The result is shown in the following figure: