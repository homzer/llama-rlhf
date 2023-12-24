import torch


class SlimLogits:
    def __init__(self, n=5, logits: torch.Tensor = None):
        self.n = n
        self.vocab_size = None
        self.batch_size = None
        self.max_seq_len = None
        self.values = None
        self.indices = None
        if logits is not None:
            self.batch_size = logits.shape[0]
            self.max_seq_len = logits.shape[1]
            self.vocab_size = logits.shape[2]
            self._set(logits)

    def _set(self, logits: torch.Tensor):
        (batch_size, seq_len, vocab_size) = logits.shape
        assert self.batch_size == batch_size
        assert self.vocab_size == vocab_size
        values, indices = torch.topk(logits, k=self.n)
        self.values = values.cpu()
        self.indices = indices.cpu()

    # def add(self, logits: torch.Tensor):
    #     """ add logits along 'max_seq_len' dim. """
    #     assert not self.start < 0
    #     (batch_size, seq_len, vocab_size) = logits.shape
    #     assert self.batch_size == batch_size
    #     assert self.vocab_size == vocab_size
    #     assert self.max_seq_len - self.start >= seq_len
    #     values, indices = torch.topk(logits, k=self.n)
    #     self.values[:, self.start: self.start + seq_len, :] = values.cpu().numpy()
    #     self.indices[:, self.start: self.start + seq_len, :] = indices.cpu().numpy()
    #     self.start += seq_len

    def extend(self, slim_logits):
        """ Batch extend. """
        if self.vocab_size is None:
            self.vocab_size = slim_logits.vocab_size
        else:
            assert self.vocab_size == slim_logits.vocab_size
        if self.max_seq_len is None:
            self.max_seq_len = slim_logits.max_seq_len
        else:
            assert self.max_seq_len == slim_logits.max_seq_len

        self.values = slim_logits.values if (
                self.values is None
        ) else torch.cat([self.values, slim_logits.values], dim=0)
        self.indices = slim_logits.indices if (
                self.indices is None
        ) else torch.cat([self.indices, slim_logits.indices], dim=0)

    def __len__(self):
        if self.values is not None and self.indices is not None:
            return len(self.values)
        return 0

    def fetch(self, i: int) -> torch.Tensor:
        assert 0 <= i < len(self), "Index out of range error"
        value = self.values[i]  # [s, n]
        index = self.indices[i]  # [s, n]
        logits = torch.full((self.max_seq_len, self.vocab_size), fill_value=1e-5, dtype=torch.float32)
        for j in range(self.max_seq_len):
            logits[j, index[j]] = value[j]
        return logits  # [s, v]
