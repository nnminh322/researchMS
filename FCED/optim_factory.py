# optim_factory.py

import torch
from sam_family import SAMFamily


def build_optimizer(model, args):
    """
    Factory tạo optimizer theo args.optim:

        args.optim in {"adamw", "sam", "asam", "fsam"}

    Cần các tham số trong args:
        - lr
        - weight_decay  (tuỳ bạn)
        - rho           (cho SAM-family)
        - fisher_topk   (cho FSAM, optional)
        - fisher_ema    (cho FSAM, optional)
    """

    params = model.parameters()

    if args.optim.lower() == "adamw":
        return torch.optim.AdamW(
            params,
            lr=args.lr,
            weight_decay=getattr(args, "weight_decay", 0.0),
        )

    elif args.optim.lower() in ("sam", "asam", "fsam"):
        mode = args.optim.lower()

        return SAMFamily(
            params,
            base_optimizer=torch.optim.AdamW,
            lr=args.lr,
            weight_decay=getattr(args, "weight_decay", 0.0),
            rho=getattr(args, "rho", 0.05),
            mode=mode,
            fisher_topk=getattr(args, "fisher_topk", 0.2),
            fisher_ema=getattr(args, "fisher_ema", 0.9),
        )

    else:
        raise ValueError(f"Unknown optimizer: {args.optim}")


def fisher_warmup_and_train(
    model,
    train_loader,
    args,
    device="cuda",
):
    """
    Ví dụ pipeline:
        - Warmup vài epoch để ước lượng Fisher
        - Build mask
        - Train chính với FSAM

    Bạn có thể chỉnh lại logic epoch/batch cho phù hợp.
    """
    optimizer = build_optimizer(model, args)

    model.to(device)

    # Warmup Fisher (giả sử args.optim == "fsam")
    if args.optim.lower() == "fsam":
        model.train()
        num_warmup_steps = getattr(args, "fisher_warmup_steps", 100)

        step_count = 0
        for batch in train_loader:
            if step_count >= num_warmup_steps:
                break

            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()

            # update Fisher từ grad hiện tại
            optimizer.update_fisher()

            # bước tối ưu "bình thường" nếu muốn (ở đây có thể bỏ qua hoặc dùng AdamW)
            optimizer.base_optimizer.step()

            step_count += 1

        # Sau warmup, build mask
        optimizer.build_fisher_mask()

    # Sau đó train chính (SAM / FSAM / ASAM) vẫn dùng closure như bình thường
    # (đoạn này bạn tự tích hợp vào loop hiện tại của bạn)
    return optimizer
