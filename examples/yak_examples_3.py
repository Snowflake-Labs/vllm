from vllm import LLM, SamplingParams
# Sample prompts.
prompts = [
    #"Hello, my name is",
    "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0)
sampling_params = SamplingParams(temperature=0.0,  min_tokens=10000, max_tokens=10000)
# Create an LLM.
yak_model_path = "facebook/opt-125m"
yak_model_path = "/data-fast/yak2b-25B-500B-phase2-instruct-v4-hf"
yak_model_path = "/data-fast/small-yak2c-long-seq-bookonly-eval-ckpt/conv1-fast-quant/2500/"
yak_model_path = "/data-fast/yak2b-25B-500B-phase2-instruct-v4-hf/"
#yak_model_path = "/checkpoint/yak2b-25B-500B-phase2-instruct-v4-hf1"
llm = LLM(model=yak_model_path, 
          quantization="deepspeedfp",
          tensor_parallel_size=8,
          disable_custom_all_reduce=True,
          enforce_eager =True, 
          )
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
#import pdb
#pdb.set_trace()
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
