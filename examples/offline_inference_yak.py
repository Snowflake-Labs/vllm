from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
     "The capital of France is",
     "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
# yak_model_path = "/checkpoint/700M_tulu_yak_tp1"
yak_model_path = "/checkpoint/yak2b-25B-500B-phase2-instruct-v4-hf1"
llm = LLM(model=yak_model_path, 
          quantization="yq",
          tensor_parallel_size=8)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.

outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
