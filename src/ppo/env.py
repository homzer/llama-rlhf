from typing import List

import numpy as np

from src.generator import GeneratorForVerifier
from src.modeling.llama_lora import LoraLlamaVerifier
from src.tokenizer import LlamaTokenizer


class LlamaRewardEnv:
    def __init__(self, verifier: LoraLlamaVerifier, tokenizer: LlamaTokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.generator = GeneratorForVerifier(verifier, tokenizer, max_seq_len)

    def step(self, obs: List[str], actions: np.array, action_masks: np.array) -> List[List[float]]:
        outputs = []
        for action, action_mask in zip(actions, action_masks):
            outputs.append(self.tokenizer.decode(action[action_mask].tolist()))
        # TODO
        print(outputs[0])
        return self.generator.forward(obs, outputs).tokens_rewards
