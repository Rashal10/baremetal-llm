import tempfile
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from baremetal_llm.foundation import (
    CharDataLoader,
    CharTokenizer,
    GroupedAttention,
    ModernLM,
    TinyLM,
    train_step,
)
from baremetal_llm.scaling import InstructionCollator, MixtureOfExperts, load_instruction_data
from baremetal_llm.utils.checkpoints import save_checkpoint
from baremetal_llm.utils.device import get_device
from baremetal_llm.utils.paths import data_path, repo_root

DEVICE = get_device(False)
TOK = CharTokenizer()

PLOT_STYLE = {
    "figure.facecolor": "#fafafa",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#e0e0e0",
    "axes.labelcolor": "#424242",
    "axes.titleweight": "bold",
    "grid.color": "#eeeeee",
    "font.size": 10,
}

CUSTOM_CSS = """
.gradio-container { font-family: 'Inter', system-ui, sans-serif !important; }
#header { text-align: center; padding: 1.5rem 0 0.5rem; }
#header h1 { font-size: 1.75rem; font-weight: 700; margin: 0; color: #1a237e; }
#header p { color: #616161; margin: 0.5rem 0 0; }
.footer { text-align: center; color: #9e9e9e; font-size: 0.85rem; margin-top: 1rem; }
.tab-nav button { font-weight: 500 !important; }
"""


def _ckpt_dir() -> Path:
    local = repo_root() / "parts" / "part_2" / "runs" / "tiny-lm"
    if local.parent.parent.exists():
        local.mkdir(parents=True, exist_ok=True)
        return local
    cache = Path(tempfile.gettempdir()) / "baremetal-llm" / "tiny-lm"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _fig_path(fig) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return tmp.name


def _style_axes(ax, title: str):
    ax.set_title(title, pad=10)
    ax.grid(True, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_color("#e0e0e0")


def _default_ckpt(name: str) -> Path | None:
    p = repo_root() / "parts" / name
    if p.exists():
        return p
    fallback = _ckpt_dir() / "model.pt"
    return fallback if fallback.exists() else None


def tab_generate(prompt: str, max_tokens: int, model_kind: str, ckpt_path: str):
    if not prompt.strip():
        return "Please enter a prompt."

    path = Path(ckpt_path.strip()) if ckpt_path.strip() else None
    if path is None or not path.exists():
        if model_kind == "tiny":
            path = _default_ckpt("part_2/runs/tiny-lm/model.pt")
        else:
            path = _default_ckpt("part_6/runs/sft-demo/model_last.pt") or _default_ckpt(
                "part_3/runs/modern-lm/model.pt"
            )

    if path is None or not path.exists():
        return "No checkpoint available. Train a model on the **Train** tab first."

    payload = torch.load(path, map_location=DEVICE, weights_only=False)
    kind = payload.get("model_type", "tiny" if model_kind == "tiny" else "modern")
    config = payload.get("config", {"vocab": 256, "ctx_len": 128, "n_layers": 2, "n_heads": 2, "dim": 64})

    model_cls = TinyLM if kind == "tiny" else ModernLM
    model = model_cls(**config)
    model.load_state_dict(payload["model"])
    model.to(DEVICE).eval()

    ids = TOK.encode(prompt).unsqueeze(0).to(DEVICE)
    out = model.generate(ids, max_tokens=int(max_tokens))
    return TOK.decode(out[0])


def tab_train(steps: int):
    steps = int(steps)
    data = CharDataLoader(str(data_path()), ctx_len=32)
    model = TinyLM(vocab=256, ctx_len=32, n_layers=2, n_heads=2, dim=64).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=3e-3)

    losses = []
    for _ in range(steps):
        x, y = data.get_batch("train", 8, DEVICE)
        losses.append(train_step(model, x, y, opt))

    ckpt = _ckpt_dir() / "model.pt"
    save_checkpoint(ckpt, model, {"model_type": "tiny"})

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6, 2.8))
        ax.plot(losses, color="#3949AB", linewidth=2)
        _style_axes(ax, "Training loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cross-entropy")
        img = _fig_path(fig)

    sample = model.generate(TOK.encode("Transformers ").unsqueeze(0).to(DEVICE), max_tokens=40)
    return img, TOK.decode(sample[0]), str(ckpt)


def tab_attention():
    torch.manual_seed(0)
    x = torch.randn(1, 12, 64)
    attn = GroupedAttention(64, 4)
    with torch.no_grad():
        _, w = attn(x)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(w[0, 0].cpu().numpy(), cmap="magma", aspect="auto")
        _style_axes(ax, "Causal attention, head 0")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        fig.colorbar(im, ax=ax, fraction=0.046)
        img = _fig_path(fig)
    return img


def tab_moe():
    torch.manual_seed(1)
    moe = MixtureOfExperts(dim=64, n_experts=8, k=2)
    x = torch.randn(4, 16, 64)
    _, aux = moe(x)

    with torch.no_grad():
        flat = x.reshape(-1, 64)
        idx, _, _ = moe.router(flat)
        counts = torch.bincount(idx[:, 0], minlength=8).float()

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6, 2.8))
        ax.bar(range(8), counts.numpy(), color="#5C6BC0", edgecolor="#3949AB")
        _style_axes(ax, f"Expert routing (aux loss = {aux.item():.3f})")
        ax.set_xlabel("Expert index")
        ax.set_ylabel("Token assignments")
        img = _fig_path(fig)
    return img


def tab_align(prompt: str, steps: int):
    if not prompt.strip():
        prompt = "What is the first prime number?"

    items = load_instruction_data(use_fallback=True)[:4]
    collator = InstructionCollator(ctx_len=64)
    x, y = collator.collate([(d.prompt, d.response) for d in items])
    x, y = x.to(DEVICE), y.to(DEVICE)

    base = ModernLM(vocab=256, ctx_len=64, n_layers=2, n_heads=2, dim=64).to(DEVICE)
    tuned = ModernLM(vocab=256, ctx_len=64, n_layers=2, n_heads=2, dim=64).to(DEVICE)
    tuned.load_state_dict(base.state_dict())
    opt = optim.AdamW(tuned.parameters(), lr=2e-3)

    for _ in range(int(steps)):
        opt.zero_grad()
        _, loss, _ = tuned(x, y)
        loss.backward()
        opt.step()

    def gen(model, p):
        model.eval()
        ids = TOK.encode(p[:80]).unsqueeze(0).to(DEVICE)
        out = model.generate(ids, max_tokens=40, temp=0.8)
        return TOK.decode(out[0])

    return gen(base, prompt), gen(tuned, prompt)


def build_ui():
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        button_primary_background_fill="#3949AB",
        button_primary_background_fill_hover="#303F9F",
        block_title_text_weight="600",
        block_label_text_weight="500",
    )

    with gr.Blocks(
        title="Baremetal LLM",
        theme=theme,
        css=CUSTOM_CSS,
        fill_height=True,
    ) as demo:
        gr.HTML(
            """
            <div id="header">
              <h1>Baremetal LLM</h1>
              <p>From-scratch transformers, MoE, SFT &amp; RLHF in PyTorch</p>
            </div>
            """
        )

        with gr.Tabs():
            with gr.Tab("Generate"):
                gr.Markdown("Run inference from a trained checkpoint. Use **Train** first if none exists.")
                with gr.Row():
                    prompt = gr.Textbox(
                        label="Prompt",
                        value="Transformers ",
                        lines=2,
                        scale=2,
                    )
                    max_tok = gr.Slider(10, 120, value=50, step=5, label="Max tokens")
                with gr.Row():
                    kind = gr.Radio(["tiny", "modern"], value="tiny", label="Architecture")
                    ckpt = gr.Textbox(
                        label="Checkpoint path (optional)",
                        placeholder="auto-detect from Train tab",
                        scale=2,
                    )
                out_gen = gr.Textbox(label="Generated text", lines=5)
                gr.Button("Generate", variant="primary").click(
                    tab_generate, [prompt, max_tok, kind, ckpt], out_gen
                )

            with gr.Tab("Train"):
                gr.Markdown("Train a character-level **TinyLM** on the bundled corpus (~1 min on CPU).")
                steps = gr.Slider(20, 150, value=60, step=10, label="Training steps")
                with gr.Row():
                    loss_img = gr.Image(label="Loss curve", type="filepath")
                    with gr.Column():
                        sample_out = gr.Textbox(label="Sample generation", lines=4)
                        ckpt_out = gr.Textbox(label="Checkpoint saved to")
                gr.Button("Start training", variant="primary").click(
                    tab_train, steps, [loss_img, sample_out, ckpt_out]
                )

            with gr.Tab("Attention"):
                gr.Markdown("Visualize causal self-attention weights from a multi-head layer.")
                attn_img = gr.Image(label="Attention heatmap", type="filepath")
                gr.Button("Render heatmap", variant="primary").click(tab_attention, None, attn_img)

            with gr.Tab("MoE"):
                gr.Markdown("Expert routing distribution from a Mixture-of-Experts layer.")
                moe_img = gr.Image(label="Routing histogram", type="filepath")
                gr.Button("Render histogram", variant="primary").click(tab_moe, None, moe_img)

            with gr.Tab("Align"):
                gr.Markdown("Compare base vs. quick SFT on instruction data.")
                p = gr.Textbox(label="Prompt", value="What is the first prime number?")
                sft_steps = gr.Slider(10, 80, value=30, step=5, label="SFT steps")
                with gr.Row():
                    base_out = gr.Textbox(label="Base model", lines=4)
                    tuned_out = gr.Textbox(label="After SFT", lines=4)
                gr.Button("Compare outputs", variant="primary").click(
                    tab_align, [p, sft_steps], [base_out, tuned_out]
                )

        gr.HTML(
            """
            <div class="footer">
              <a href="https://github.com/Rashal10/baremetal-llm" target="_blank">GitHub</a>
              &nbsp;·&nbsp;
              <a href="https://rashal10.github.io/baremetal-llm/" target="_blank">Docs</a>
            </div>
            """
        )

    return demo


if __name__ == "__main__":
    build_ui().launch()
