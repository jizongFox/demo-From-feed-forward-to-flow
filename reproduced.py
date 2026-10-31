import torch
from tqdm import tqdm

from modules import create_dataset, FeedForwardNet

ffn = FeedForwardNet(input_dim=8).cuda()
T_max = 2000
r = 0.7
optimizer = torch.optim.AdamW(ffn.parameters(), lr=1e-2, weight_decay=0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
with tqdm(range(T_max), total=T_max) as indicator:
    for i in indicator:
        x0, cr, *_ = create_dataset(1000, K=8, noise=0.1, r=r)
        optimizer.zero_grad()

        predict = ffn(cr)
        loss = (x0 - predict).pow(2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        indicator.set_postfix({"loss": loss.item()})

x0, cr, mode, *_ = create_dataset(5000, K=8, noise=0.1, r=r)
ffn.eval()
with torch.no_grad():
    predict = ffn(cr)


import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2)
axes[0].scatter(x0.cpu().numpy()[:, 0], x0.cpu().numpy()[:, 1], c=mode.cpu())
axes[0].set_xlim(-3, 3)
axes[0].set_ylim(-3, 3)
axes[0].set_aspect("equal", adjustable="box")
axes[1].scatter(predict.cpu()[:, 0], predict.cpu()[:, 1], c=mode.cpu())
axes[1].set_xlim(-3, 3)
axes[1].set_ylim(-3, 3)
axes[1].set_aspect("equal", adjustable="box")
fig.tight_layout()
fig.savefig("feedforward_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# %%
from modules import FlowMatchingNet

T_max=6000
fmnet = FlowMatchingNet(input_dim=8).cuda()
optimizer = torch.optim.AdamW(fmnet.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)


def training():
    with tqdm(range(T_max), total=T_max) as indicator:
        for i in indicator:
            optimizer.zero_grad()
            x0, cr, mode, xt, t = create_dataset(
                256, K=8, noise=0.15, r=r, max_time=0.95,
                endpoint_probability=0.1,
            )
            x0_pred = fmnet(xt=xt, t=t, c=cr)
            weight = (1 - t[..., None].clamp(max=0.95)).pow(-2)
            loss = (weight * (x0_pred - x0).pow(2).sum(dim=-1, keepdim=True)).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            indicator.set_postfix(dict(loss=loss.item()))


@torch.no_grad()
def evaluate(num_steps: int = 15, manual_mode_num:None|int=None, data=None):
    fmnet.eval()
    if data is None:
        x0, cr, mode, *_ = create_dataset(
            1000, K=8, noise=0.15, r=r, manual_model_num=manual_mode_num
        )
    else:
        x0, cr, mode = data
    x = torch.randn_like(x0)
    for i in range(num_steps):
        t = torch.full((x.shape[0],), i / num_steps, device=x.device)
        x0_pred = fmnet(xt=x, t=t, c=cr)
        x = x0_pred if i == num_steps - 1 else x + (x0_pred - x) / (num_steps - i)

    return x, x0, mode


def conditional_mean(cr):
    angles = torch.arange(8, device=cr.device) * (2 * torch.pi / 8)
    centers = 2 * torch.stack((angles.cos(), angles.sin()), dim=-1)
    sigma = 2.5 + r * (0.08 - 2.5)
    return (cr / sigma**2).softmax(dim=-1) @ centers


import matplotlib.pyplot as plt

fig, axes = plt.subplots(6, 5, figsize=(13, 15))
titles = ("Ground truth", "Mean", "Feed-forward", "Flow-1", "Flow-15")
for row, reliability in enumerate((0.4, 0.9, 1.0)):
    r = reliability
    fmnet = FlowMatchingNet(input_dim=8).cuda()
    optimizer = torch.optim.AdamW(fmnet.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    training()
    for block, fixed_mode in enumerate((None, 0)):
        data = create_dataset(
            1000, K=8, noise=0.15, r=r, manual_model_num=fixed_mode
        )[:3]
        gt, cr, mode = data
        flow1, _, _ = evaluate(1, data=data)
        flow15, _, _ = evaluate(15, data=data)
        mean = conditional_mean(cr)
        feedforward = ffn(cr).detach()
        values = (gt, mean, feedforward, flow1, flow15)
        for col, (ax, points) in enumerate(zip(axes[row * 2 + block], values)):
            ax.scatter(points[:, 0].cpu(), points[:, 1].cpu(), c=mode.cpu(), s=3)
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col])
        axes[row * 2 + block, 0].set_ylabel(
            f"r={r}\n{'all modes' if fixed_mode is None else 'fixed mode'}"
        )

fig.tight_layout()
fig.savefig("flow_matching_figure1.png", dpi=200, bbox_inches="tight")
fig.savefig("flow_matching_figure1.pdf", bbox_inches="tight")
plt.close(fig)
