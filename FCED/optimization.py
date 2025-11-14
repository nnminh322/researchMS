# sam_family.py

import torch
from torch.optim import Optimizer


class SAMFamily(Optimizer):
    """
    SAM / ASAM / FSAM trong một class.

    mode:
        - "sam"  : SAM gốc (sharpness-aware minimization)
        - "asam" : Adaptive SAM (scale theo |w|)
        - "fsam" : Fisher-SAM (SAM + Fisher mask, chỉ perturb top-k hướng "sắc")

    base_optimizer:
        - class optimizer của PyTorch, ví dụ torch.optim.AdamW

    Cách dùng (ví dụ):
        optimizer = SAMFamily(
            model.parameters(),
            base_optimizer=torch.optim.AdamW,
            lr=1e-4,
            rho=0.05,
            mode="sam",        # hoặc "asam", "fsam"
        )

        def closure():
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
    """

    def __init__(
        self,
        params,
        base_optimizer,
        rho: float = 0.05,
        mode: str = "sam",
        adaptive: bool | None = None,
        fisher_topk: float = 0.2,
        fisher_ema: float = 0.9,
        **kwargs,
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert mode in ("sam", "asam", "fsam"), f"Invalid mode: {mode}"
        if adaptive is None:
            adaptive = (mode == "asam")

        defaults = dict(
            rho=rho,
            mode=mode,
            adaptive=adaptive,
            fisher_topk=fisher_topk,
            fisher_ema=fisher_ema,
            **kwargs,
        )
        super().__init__(params, defaults)

        # Khởi tạo base optimizer (AdamW, SGD,...)
        # Lưu ý: base_optimizer nhận param_groups, không phải params thô
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        # Đồng bộ param_groups cho đúng tham chiếu
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    # ------------------------------------------------------------------
    # SAM core
    # ------------------------------------------------------------------

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """
        Bước 1: đi từ w -> w + e(w) (climb lên local max).
        """
        grad_norm = self._grad_norm() + 1e-12

        for group in self.param_groups:
            rho = group["rho"]
            mode = group["mode"]
            adaptive = group["adaptive"]

            scale = rho / grad_norm

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                state["old_p"] = p.data.clone()

                # weight theo adaptive (ASAM) hoặc không (SAM)
                w = torch.abs(p) if adaptive else 1.0

                # nếu FSAM thì nhân thêm fisher_mask (nếu đã build)
                if mode == "fsam":
                    mask = state.get("fisher_mask", None)
                    if mask is not None:
                        w = w * mask

                e_w = w * p.grad * scale.to(p)
                p.add_(e_w)  # w <- w + e(w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """
        Bước 2: quay lại w, apply bước update thật sự của base_optimizer.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # quay về w

        # Bước tối ưu thực sự
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        Chuẩn SAM: cần closure làm full forward + backward.
        Gọi:
            loss = closure()
        2 lần, một lần tại w, một lần tại w + e(w).
        """
        assert closure is not None, "SAM-family requires a closure"
        closure = torch.enable_grad()(closure)

        # 1) forward/backward tại w, grad lưu trong p.grad
        #    -> first_step dịch sang w + e(w)
        self.first_step(zero_grad=True)

        # 2) forward/backward tại w + e(w) để có grad mới
        closure()

        # 3) second_step: quay về w, apply base_optimizer
        self.second_step()

    def _grad_norm(self) -> torch.Tensor:
        """
        ||g|| với trọng số thích hợp (adaptive / fisher mask).
        """
        shared_device = self.param_groups[0]["params"][0].device
        norms = []

        for group in self.param_groups:
            adaptive = group["adaptive"]
            mode = group["mode"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                w = torch.abs(p) if adaptive else 1.0
                if mode == "fsam":
                    mask = self.state[p].get("fisher_mask", None)
                    if mask is not None:
                        w = w * mask

                norms.append((w * p.grad).norm(p=2).to(shared_device))

        if not norms:
            return torch.tensor(0.0, device=shared_device)

        return torch.norm(torch.stack(norms), p=2)

    def load_state_dict(self, state_dict):
        """
        Override để đảm bảo base_optimizer.param_groups trùng với self.param_groups.
        """
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    # ------------------------------------------------------------------
    # FSAM: Fisher estimation + mask
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_fisher(self):
        """
        Gọi sau một backward trên loss chính (không SAM),
        dùng grad hiện tại để update Fisher (EMA) cho từng tham số.

        Thường:
            loss = ...
            loss.backward()
            optimizer.update_fisher()
        """
        for group in self.param_groups:
            ema = group["fisher_ema"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                g2 = (p.grad.detach() ** 2)

                if "fisher" not in state:
                    state["fisher"] = g2.clone()
                else:
                    state["fisher"].mul_(ema).add_(g2, alpha=1 - ema)

    @torch.no_grad()
    def build_fisher_mask(self):
        """
        Gọi sau khi đã tích luỹ Fisher (ví dụ sau warmup).
        Sinh mask nhị phân (0/1) trên từng tham số dựa trên top-k giá trị Fisher.
        """
        all_fisher_flat = []

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "fisher" in state:
                    all_fisher_flat.append(state["fisher"].view(-1))

        if not all_fisher_flat:
            return

        all_fisher_flat = torch.cat(all_fisher_flat)
        topk_ratio = self.defaults.get("fisher_topk", 0.2)

        if topk_ratio <= 0.0:
            return

        k = int(len(all_fisher_flat) * topk_ratio)
        if k < 1:
            return

        # Ngưỡng Fisher: top-k
        thresh = torch.topk(all_fisher_flat, k)[0][-1]

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                fisher = state.get("fisher", None)
                if fisher is None:
                    continue
                mask = (fisher >= thresh).float()
                state["fisher_mask"] = mask

    @torch.no_grad()
    def set_mode(self, mode: str):
        """
        Đổi mode sau khi đã khởi tạo optimizer.
        Ví dụ: warmup ở mode="sam", sau đó chuyển sang "fsam".
        """
        assert mode in ("sam", "asam", "fsam")
        for group in self.param_groups:
            group["mode"] = mode
            if mode == "asam":
                group["adaptive"] = True
            elif mode == "sam":
                group["adaptive"] = False
            # fsam: adaptive tùy bạn set từ đầu


# ----------------------------------------------------------------------
# TRAM helper: Trust-Region loss trong function space
# ----------------------------------------------------------------------

def tram_loss(
    logits: torch.Tensor,
    anchor_logits: torch.Tensor,
    tr_type: str = "mse",
) -> torch.Tensor:
    """
    Trust-region loss trong function space.

    tr_type:
        - "mse": MSE giữa logits
        - "kl" : KL(anchor || current) trên phân phối softmax
    """
    if tr_type == "mse":
        return torch.nn.functional.mse_loss(logits, anchor_logits)

    elif tr_type == "kl":
        # KL(anchor || current)
        log_p = torch.nn.functional.log_softmax(anchor_logits, dim=-1)
        log_q = torch.nn.functional.log_softmax(logits, dim=-1)
        p = log_p.exp()
        # sum_i p_i (log p_i - log q_i)
        return torch.sum(p * (log_p - log_q), dim=-1).mean()

    else:
        raise ValueError(f"Unknown tr_type: {tr_type}")


def make_tram_closure(
    model,
    optimizer: SAMFamily,
    batch,
    loss_fn,
    anchor_model=None,
    lambda_tr: float = 0.0,
    tr_type: str = "mse",
    device: str | torch.device = "cuda",
):
    """
    Tạo closure cho TRAM: loss_task + lambda_tr * tram_loss.

    - model: mô hình hiện tại
    - optimizer: SAMFamily (hoặc optimizer nào cần closure)
    - batch: (inputs, labels, ...) – bạn tự unpack
    - loss_fn: hàm loss chính (CE, focal, ...)
    - anchor_model: mô hình anchor (pretrained / EMA); nếu None hoặc lambda_tr=0 thì chỉ có loss_task
    - lambda_tr: hệ số trust-region
    - tr_type: "mse" hoặc "kl"
    """

    inputs, labels = batch  # tuỳ dataset bạn sửa lại cho phù hợp
    inputs = inputs.to(device)
    labels = labels.to(device)

    def closure():
        optimizer.zero_grad()

        # forward hiện tại
        logits = model(inputs)
        loss_task = loss_fn(logits, labels)

        if anchor_model is not None and lambda_tr > 0.0:
            with torch.no_grad():
                anchor_logits = anchor_model(inputs)
            loss_tr = tram_loss(logits, anchor_logits, tr_type=tr_type)
            loss = loss_task + lambda_tr * loss_tr
        else:
            loss = loss_task

        loss.backward()
        return loss

    return closure


# ----------------------------------------------------------------------
# Selective SAM helper
# ----------------------------------------------------------------------

def selective_sam_step(
    optimizer: SAMFamily,
    closure_sam,
    closure_plain=None,
):
    """
    Selective SAM:
        - Bước 1: SAM trên một phần loss (Led, Lkt,...): closure_sam
        - Bước 2: plain step (base_optimizer) trên loss khác (Lr, Ld, ...) nếu cần

    Cách dùng ví dụ (continual):

        # 1) SAM trên loss hiện tại (Led + Lkt)
        def closure_sam():
            optimizer.zero_grad()
            loss = loss_led + loss_lkt
            loss.backward()
            return loss

        # 2) Plain trên loss replay (Lr + Ld)
        def closure_plain():
            optimizer.zero_grad()
            loss = loss_lr + loss_ld
            loss.backward()
            return loss

        selective_sam_step(optimizer, closure_sam, closure_plain)
    """

    # SAM step (hai lần forward/backward)
    optimizer.step(closure_sam)

    # Plain step (không SAM) nếu có
    if closure_plain is not None:
        optimizer.zero_grad()
        loss_plain = closure_plain()
        # Ở đây closure_plain đã backward rồi
        optimizer.base_optimizer.step()
        return loss_plain
