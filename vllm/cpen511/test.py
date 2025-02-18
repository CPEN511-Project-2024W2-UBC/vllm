from vllm import LLM, SamplingParams
from vllm.cpen511.swap_trace_logger import SwapTraceLogger

# Sample prompts.
prompts = [
    "The future of AI is still uncertain, but let's think about the future of AI in this problem. We are given a sequence of integers, and we need to compute the probability that a particular AI model will successfully complete a task in the future given that it has successfully completed the task in the past. The probability is defined as the expected value of a function f, which is the expected value over all possible task sequences. We have to compute the probability that the AI will successfully complete a task in the future given that it has succeeded in the past. The probability is given by the formula P(future | past) = P(future and past) / P(past).  The probability that the AI will successfully complete a task in the future given that it has succeeded in the past is given by the formula P(future | past) = P(future and past) / P(past). The probability that the AI will successfully complete a task in the future given that it has succeeded in the past is given by the formula P(future | past) = P(future and past) / P(past). The probability that the AI will successfully complete a task in the future given that it has succeeded in the past is given by the formula P(future | past) = P(future and past) / P(past). The probability that the AI will successfully complete a task in the future given that it has succeeded in the past is given by the formula P(future | past) = P(future and past) / P(past). The probability that the AI will successfully complete a task in the future given that it has succeeded in the past is given by the formula P(future | past) = P(future and past) / P(past). The probability that the AI will successfully complete a task in the future given that it has succeeded in the past is given by the formula P(future | past) = P(future and past) / P(past). The probability that the AI will successfully complete a task in the future given that it has succeeded in the past is given by the formula P(future | past) = P(future and past) / P(past). The probability that the AI will successfully complete a task in the future given that it has succeeded in the past is given by the formula P(future | past) = P(future and past) / P(past). The probability that the AI will successfully complete a task in the future given that it has succeeded in the past is given by the formula P(future | past) = P(future and past) / P(past). The probability that the AI will successfully complete a task in the future given that it has succeeded in the past is given by the formula P(future | past) = P(future and past) / P(past). The probability that the AI will successfully complete a task in the future given that it has succeeded in the past is given by the formula P(future | past) = P(future and past) / P(past). The probability that the AI will successfully complete a task in the future given that", 
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", gpu_memory_utilization=0.3, max_model_len=300)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
    
SwapTraceLogger.get_instance().get_swap_trace()