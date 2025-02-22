from vllm import LLM, SamplingParams
import pandas as pd

# read HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv, select Behavior column as prompts
prompts = pd.read_csv('vllm/cpen511/data/harmbench_behaviors_text_test.csv', nrows=1000)['Behavior'].tolist()

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95,  max_tokens=100)

# Create an LLM.
llm = LLM(model="facebook/opt-125m", gpu_memory_utilization=0.05, max_model_len=384, preemption_mode="swap", scheduling_policy='priority')
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)

# Write the outputs to a file.
with open('results.txt', 'w') as f:
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        f.write(f"Prompt: {prompt!r}, Generated text: {generated_text!r}\n")