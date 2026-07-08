# Live Demo

## Hugging Face Spaces

Deploy the Gradio app so interviewers can try it without cloning.

### One-time setup

1. Create a Hugging Face account at [huggingface.co](https://huggingface.co).
2. Create a new **Space** → SDK: **Gradio** → name: `baremetal-llm-demo`.
3. Upload the files from the `space/` folder in this repo (or connect the Space to GitHub).

### Files to upload

```
space/
  README.md          # Space metadata
  app.py             # Gradio entry point
  requirements.txt   # pip deps for the Space
```

### After deploy

Your demo URL will be:

`https://huggingface.co/spaces/Rashal10/baremetal-llm-demo`

Add it to your resume and the main README.

## Local Gradio

```bash
pip install -e ".[demo]"
python demos/app.py
```

Default: `http://127.0.0.1:7860`

## Colab

No local install needed:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rashal10/baremetal-llm/blob/main/notebooks/01_quick_demo.ipynb)

## GitHub Pages (this site)

Docs deploy automatically on push to `main` via `.github/workflows/docs.yml`.

Enable in repo **Settings → Pages → Source: GitHub Actions**.

Site URL: `https://rashal10.github.io/baremetal-llm/`
