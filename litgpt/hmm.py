import torch
import torch.nn as nn
from torch.nn.functional import log_softmax


class HMM(nn.Module):
    def __init__(
        self, n_hidden_states: int, vocab_size: int, max_seq_length: int
    ) -> None:
        self.n_hidden_states = n_hidden_states
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.em = torch.tensor(torch.randn(n_hidden_states, vocab_size))
        self.tm = torch.tensor(torch.randn(n_hidden_states, n_hidden_states))
        self.p = torch.tensor(torch.randn(n_hidden_states))

    def forward_algo(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns alpha matrix: alpha_it = P(o_1, o_2, ..., o_t, S_t = i)
        input_ids [b, s]
        """
        em = log_softmax(self.em, dim=-1)
        tm = log_softmax(self.tm, dim=-1)
        p = log_softmax(self.p, dim=-1)
        b, s = input_ids.size()
        alpha = torch.zeros(b, self.n_hidden_states, self.max_seq_length)
        # [b, h] = [b] + [h, b].T
        alpha[:, :, 0] = p + em[:, input_ids[:, 0]].T
        for t in range(1, s):
            # [b, h] = [h, b].T + lse([b, 1, h] + [h, h]) -> [b, h]
            alpha[:, :, t] = em[:, input_ids[:, t]].T + torch.logsumexp(
                alpha[:, :, t - 1].unsqueeze(1) + tm, dim=1
            )
        return alpha  # [b, h, s]

    def backward_algo(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns beta matrix: beta_it = P(o_t+1, o_t+2, ..., o_T | S_t = i)
        input_ids [b, s]
        """
        em = log_softmax(self.em, dim=-1)
        tm = log_softmax(self.tm, dim=-1)
        # p = log_softmax(self.p, dim=-1)
        b, s = input_ids.size()
        beta = torch.zeros(b, self.n_hidden_states, self.max_seq_length)
        beta[:, :, -1] = 0
        for t in range(s - 2, -1, -1):
            beta[:, :, t] = torch.logsumexp(
                beta[:, :, t + 1].unsqueeze(1)  # [b, 1, h]
                + em[:, input_ids[:, t + 1]].T.unsqueeze(
                    1
                )  # [h, b].T.unsqueeze(1) -> [b, 1, h]
                + tm,  # [h, h]
                dim=1,
            )
        return beta  # [b, h, s]

    def update_parameters(self, input_ids: torch.Tensor) -> None:
        """
        Use expectation-maximization to update parameters.
        input_ids [b, s]
        """
        n_batches = input_ids.size(0)
        for i in range(n_batches):
            x = input_ids[i].unsqueeze(0)  # [1, s]
            alpha = self.forward_algo(x).squeeze(0)  # [h, s]
            beta = self.backward_algo(x).squeeze(0)  # [h, s]

            # 1. compute gamma_it = P(S_t=i | o_1, ... , o_T), [h, s]
            gamma = alpha + beta  # [h, s]
            gamma = gamma - torch.logsumexp(gamma, dim=0)  # [h, s] - [s]

            # 2. compute delta_ijt = P(S_t=i, S_t+1=j | o_1, ... , o_T), [h, h, s]
            em = log_softmax(self.em, dim=-1)
            tm = log_softmax(self.tm, dim=-1)
            x = x.squeeze(0)  # [s]
            delta = (  # [h, h, s-1]
                alpha[:, :-1].unsqueeze(1)  # [h, 1, s-1]
                + tm.unsqueeze(2)  # [h, h, 1]
                + em[:, x[1:]].unsqueeze(0)  # [1, h, s-1]
                + beta[:, 1:].unsqueeze(0)  # [1, h, s-1]
            )
            # [h, h, s-1] - [s-1]
            delta = delta - torch.logsumexp(alpha[:, :-1] + beta[:, :-1], dim=0)

            # 3. update priors and transition matrix
            self.p = gamma[:, 0]  # [h]
            self.tm = torch.logsumexp(delta, dim=-1) - torch.logsumexp(
                gamma, dim=-1
            )  # [h, h]
            # 4. update emission matrix
            mask = torch.zeros(self.vocab_size, self.max_seq_length)
            # use x to turn into 1-hot
            mask[x, torch.arange(x.size(0))] = 1  # [v, s]

            new_em = gamma.unsqueeze(1) * mask  # [h, 1, s] * [v, s] -> [h, v, s]
            self.em = torch.logsumexp(new_em, dim=-1) - torch.logsumexp(gamma, dim=-1)

    def forward(self, input_ids: torch.Tensor, do_em: bool) -> torch.Tensor:
        """
        Returns the log-likelihood of the input_ids.
        input_ids [b, s]
        """
        alpha = self.forward_algo(input_ids)
        ll = torch.logsumexp(alpha[:, :, -1], dim=1)  # [b]
        loss = -ll.mean()
        if do_em:
            self.update_parameters(input_ids)
        return loss
