from vllm import LLM, SamplingParams
# Sample prompts.
prompts = [
    "Vaswani et al. (2017) introduced the Transformers"
    #"Hello, my name is",
    #"The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
# Create a sampling params object.i
sampling_params = SamplingParams(temperature=0.0,  min_tokens=4000, max_tokens=4000)

#sampling_params = SamplingParams(temperature=0.0, )
# Create an LLM.
yak_model_path = "facebook/opt-125m"
yak_model_path = "daryl149/llama-2-7b-hf"
#yak_model_path = "/checkpoint/yak2b-25B-500B-phase2-instruct-v4-hf1"
llm = LLM(model=yak_model_path, 
          #quantization="yq",
          enforce_eager=True,
          tensor_parallel_size=1)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
