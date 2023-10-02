import sys

from lm_eval.base import BaseLM
from lm_eval import evaluator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm import LLM, SamplingParams


class LMEvalAdaptor(BaseLM):

    def __init__(self, llm: LLM):
        super().__init__()
        self.llm = llm
        self.tokenizer = llm.get_tokenizer()
        # self.llm = AutoModelForCausalLM.from_pretrained(llm, torch_dtype=torch.half).cuda().eval()
        # self.tokenizer = AutoTokenizer.from_pretrained(llm)
        self.vocab_size = self.tokenizer.vocab_size

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return 2048

    @property
    def max_gen_toks(self):
        raise NotImplementedError

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return "cuda"

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    @torch.no_grad()
    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        batch_size, seq_len = inps.shape
        # return torch.empty(batch_size, seq_len, self.vocab_size, device=self.device)
        # return self.llm(inps).logits
        assert batch_size == 1
        logits = self.llm.get_logits(
            prompt=None,
            prompt_token_ids=inps[0].cpu().tolist(),
        )
        logits = logits[None, :seq_len, :]
        return logits

    def _model_generate(self, context, max_length, eos_token_id):
        raise NotImplementedError


if __name__ == "__main__":
    model_name = sys.argv[-1]
    quantization = "awq" if "awq" in model_name.lower() else None
    llm = LLM(model_name, trust_remote_code=True, quantization=quantization)
    # llm = model_name
    lm_eval_model = LMEvalAdaptor(llm)
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=["arc_challenge"],
        no_cache=True,
        num_fewshot=25,
    )
    print(evaluator.make_table(results))
