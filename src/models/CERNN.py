import torch
from torch import nn

from src.models.leaky_rnn_base import RNNCell_base, RNNLayer

class CERNNModel(nn.Module):
    def __init__(
        self,
        n_input,
        n_rnn,
        n_output,
        dt,
        tau,
        noise,
        w_rec_init,
        sigma_rec,
        activation,
        ce,
    ):
        super().__init__()
        self.ce = ce
        # self.visual = self.ce.sensory[0]
        # self.somatosensory = self.ce.sensory[0]
        self.motor = self.ce.motor[0]

        # self.n_visual = self.ce.duplicates[self.visual]
        # self.n_somatosensory = self.ce.duplicates[self.somatosensory]
        self.n_motor = self.ce.duplicates[self.motor]

        decay = dt / tau
        self.n_rnn = n_rnn

        rnncell = leaky_RNNCell_CERNN(
            n_input,
            n_rnn,
            activation,
            decay,
            w_rec_init,
            ce,
            True,  # bias
            noise,
            sigma_rec,
        )

        self.rnn = RNNLayer(rnncell)
        self.readout = nn.Linear(self.n_motor, n_output, bias=False)

    def init_hidden_(self, x, h0):
        max_steps = 100
        stable_count = 0
        h0 = h0.to(self.rnn.rnncell.weight_hh.device)
        for _ in range(max_steps):
            _, h_next = self.rnn(x[:2, 0:1, :], h0)
            if torch.allclose(h_next, h0, atol=0.1):
                stable_count += 1
                if stable_count >= 4:
                    return h0
            else:
                stable_count = 0
            h0 = h_next
        return h0

    def forward(self, x):
        hidden0 = torch.zeros([1, x.shape[1], self.n_rnn])
        hidden0 = self.init_hidden_(x, hidden0)
        hidden, _ = self.rnn(x, hidden0)

        start_idx = self.ce.area2idx[self.motor]
        end_idx = start_idx + self.n_motor
        # Select the last n_motor units from the hidden state for output. n_motor assumed to be last 10 units for now
        selected_hidden = hidden[:, :, start_idx:end_idx]

        output = self.readout(selected_hidden)
        return output, hidden

class leaky_RNNCell_CERNN(RNNCell_base):
    def __init__(
        self,
        input_size,
        hidden_size,
        nonlinearity,
        decay,
        w_rec_init,
        ce,
        bias=True,
        noise=True,
        sigma_rec=0.05,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            nonlinearity=nonlinearity,
            w_rec_init=w_rec_init,
            noise=noise,
            bias=bias,
            decay=decay,
            sigma_rec=sigma_rec,
        )
        if ce.mask_within_area_weights == True:
            self.mask_within_area_weights = True
            self.register_buffer(
                "intra_area_mask", torch.zeros(self.hidden_size, self.hidden_size)
            )
            self.intra_area_mask = torch.Tensor(1 - ce.area_mask)
        else:  # False
            self.mask_within_area_weights = False
        self.zero_weights_thres = ce.zero_weights_thres
        self.visual = ce.sensory[0]
        self.somatosensory = ce.sensory[1]

        self.n_visual = ce.duplicates[self.visual]
        self.n_somatosensory = ce.duplicates[self.somatosensory]

        v1_start = ce.area2idx[self.visual]
        v1_end = v1_start + self.n_visual
        print("visual start", v1_start, "visual end", v1_end)

        s1_start = ce.area2idx[self.somatosensory]
        s1_end = s1_start + self.n_somatosensory
        print("somatosensory start", s1_start, "somatosensory end", s1_end)

        self.register_buffer("mask_s1", torch.zeros(self.hidden_size, self.input_size))
        self.mask_s1[s1_start:s1_end, 1:3] = 1

        self.register_buffer("mask_v1", torch.zeros(self.hidden_size, self.input_size))
        self.mask_v1[v1_start:v1_end, 3:5] = 1
        self.mask_v1[v1_start:v1_end, 0] = 1

        self.register_buffer(
            "mask_taskid", torch.zeros(self.hidden_size, self.input_size)
        )
        self.mask_taskid[:, 5:-1] = 1

    def get_sensory_ind(self):
        proj_mask = self.mask_s1 + self.mask_v1
        sensory_ind = torch.where(
            proj_mask.any(dim=1), torch.tensor(1), torch.tensor(0)
        )
        return sensory_ind

    def forward(self, input, hidden):
        somatosensory = input @ (self.weight_ih * self.mask_s1).t()
        visual = input @ (self.weight_ih * self.mask_v1).t()
        task_id = input @ (self.weight_ih * self.mask_taskid).t()
        input = somatosensory + visual + task_id

        weights_mask = None

        if self.mask_within_area_weights == True:
            weights_mask = self.intra_area_mask
            # print("mask_within_area_weights", weights_mask)

        if self.zero_weights_thres > 0:
            nonzero_ind = torch.where(
                torch.abs(self.weight_hh) > self.zero_weights_thres
            )
            nonzero_mask = torch.zeros(
                self.hidden_size, self.hidden_size, device=self.weight_hh.device
            )
            nonzero_mask[nonzero_ind] = 1

            if weights_mask is None:
                weights_mask = nonzero_mask
            else:
                weights_mask = torch.logical_and(weights_mask, nonzero_mask).to(
                    dtype=torch.int8
                )

        out = self.leaky_rnn_step(input, hidden, weights_mask)
        return out