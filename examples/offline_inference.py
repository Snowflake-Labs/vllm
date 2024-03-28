from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
     "The president of the United States is",
     "The capital of France is",
     "The future of AI is",
]
# Create a sampling params object.
#sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
sampling_params = SamplingParams(temperature=0)


yak_model_path = "/shared/finetuning/outputs/checkpoint/hf_ckpts/hao_ckpt/700M_tulu_yak_tp1"
yak_model_path = "/data-fast/yak2b-25B-500B-phase2-instruct-v4-hf"
#yak_model_path = "/home/yak/700M_tulu_yak_tp1"
#yak_model_path = "/shared/finetuning/outputs/checkpoint/hf_ckpts/hao_ckpt/tuluv2-2b-lr-cosine-2e-5-2e-6-adam-0.9-0.99-1e-8-bs-128-gpus-32-mp-1-pp-1-ep-64-mlc-0.01-cap-1.0-drop-false"
#yak_model_path = "/shared/finetuning/outputs/checkpoint/hf_ckpts/hao_ckpt/tulu-slimorca-sharegpt-magicoder-math-2b-step3000"
# Create an LLM.
# llm = LLM(model="mistralai/Mixtral-8x7B-v0.1", enforce_eager=True, tensor_parallel_size=2)
# llm = LLM(model=yak_model_path, enforce_eager=True, tensor_parallel_size=8)
llm = LLM(model=yak_model_path, 
          enforce_eager=True,
          quantization="yq",
          tensor_parallel_size=8)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
#import pdb; pdb.set_trace()
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
