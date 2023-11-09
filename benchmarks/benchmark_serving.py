"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    (vLLM backend)
    python -m vllm.entrypoints.api_server \
        --model <your_model> --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_hf_server.sh <your_model>

    (MII backend)
    python launch_mii_server.py --model <your_model>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --tokenizer <your_model> --dataset <target_dataset> \
        --request-rate <request_rate>
"""
import argparse
import asyncio
import json
import random
import time
from typing import AsyncGenerator, List, Optional, Tuple

import aiohttp
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []


def sample_requests(
    dataset_path: str,
    num_requests: int,
    uniform_output_len: Optional[int],
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    if uniform_output_len is not None and uniform_output_len < 4:
        raise ValueError("Too small output_len.")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        if uniform_output_len is None:
            output_len = len(completion_token_ids[i])
        else:
            output_len = uniform_output_len
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            # This is because TGI causes errors when the input or output length
            # is too short.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_http_request(
    backend: str,
    api_url: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
) -> None:
    request_start_time = time.perf_counter()

    headers = {"User-Agent": "Benchmark Client"}
    if backend == "vllm":
        pload = {
            "prompt": prompt,
            "n": 1,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
            "temperature": 0.0 if use_beam_search else 1.0,
            "top_p": 1.0,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": False,
        }
    elif backend == "tgi":
        assert not use_beam_search
        params = {
            "best_of": best_of,
            "max_new_tokens": output_len,
            "do_sample": True,
        }
        pload = {
            "inputs": prompt,
            "parameters": params,
        }
    else:
        raise ValueError(f"Unknown backend: {backend}")

    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(api_url, headers=headers,
                                    json=pload) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            output = json.loads(output)

            # Re-send the request if it failed.
            if "error" not in output:
                break

    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency))


async def send_mii_request(
    mii_client,
    prompt: str,
    prompt_len: int,
    output_len: int,
) -> None:
    request_start_time = time.perf_counter()
    # NOTE(woosuk): This uses an internal API of MII, which may change in the
    # future.
    await mii_client._request_async_response({"query": prompt},
                                             max_new_tokens=output_len)
    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency))


async def benchmark(
    backend: str,
    api_url_or_model: str,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
) -> None:
    tasks: List[asyncio.Task] = []
    if backend in ["vllm", "tgi"]:
        async for request in get_request(input_requests, request_rate):
            prompt, prompt_len, output_len = request
            task = asyncio.create_task(
                send_http_request(backend, api_url_or_model, prompt,
                                  prompt_len, output_len, best_of,
                                  use_beam_search))
            tasks.append(task)
    elif backend == "mii":
        assert best_of == 1
        assert not use_beam_search

        import mii
        mii_client = mii.client(api_url_or_model)
        async for request in get_request(input_requests, request_rate):
            prompt, prompt_len, output_len = request
            task = asyncio.create_task(
                send_mii_request(mii_client, prompt, prompt_len, output_len))
            tasks.append(task)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    await asyncio.gather(*tasks)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.model is None:
        args.model = args.tokenizer
    elif args.tokenizer is None:
        args.tokenizer = args.model

    api_url = f"http://{args.host}:{args.port}/generate"
    if args.backend in ["vllm", "tgi"]:
        url_or_model = api_url
    elif args.backend == "mii":
        url_or_model = args.model
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    input_requests = sample_requests(args.dataset, args.num_prompts,
                                     args.output_len, tokenizer)

    benchmark_start_time = time.perf_counter()
    asyncio.run(
        benchmark(args.backend, url_or_model, input_requests, args.best_of,
                  args.use_beam_search, args.request_rate))
    benchmark_end_time = time.perf_counter()
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {args.num_prompts / benchmark_time:.2f} requests/s")

    # Compute the latency statistics.
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.2f} s")
    avg_per_token_latency = np.mean([
        latency / (prompt_len + output_len)
        for prompt_len, output_len, latency in REQUEST_LATENCY
    ])
    print(f"Average latency per token: {avg_per_token_latency:.2f} s")
    avg_per_output_token_latency = np.mean(
        [latency / output_len for _, output_len, latency in REQUEST_LATENCY])
    print("Average latency per output token: "
          f"{avg_per_output_token_latency:.2f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument("--backend",
                        type=str,
                        default="vllm",
                        choices=["vllm", "tgi", "mii"])
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Path to the dataset.")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Length of the generated sequence.")
    parser.add_argument("--model",
                        type=str,
                        default=None,
                        help="Name or path of the model.")
    parser.add_argument("--tokenizer",
                        type=str,
                        required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument("--best-of",
                        type=int,
                        default=1,
                        help="Generates `best_of` sequences per prompt and "
                        "returns the best one.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--request-rate",
                        type=float,
                        default=float("inf"),
                        help="Number of requests per second. If this is inf, "
                        "then all the requests are sent at time 0. "
                        "Otherwise, we use Poisson process to synthesize "
                        "the request arrival times.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    args = parser.parse_args()
    main(args)
