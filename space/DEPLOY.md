# Deploy to Hugging Face Spaces

## Option A: connect GitHub (recommended)

1. Push this repo to GitHub.
2. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
3. Name: `baremetal-llm-demo`, SDK: **Gradio**.
4. Under **Space files**, choose **Link to a GitHub repository**.
5. Repo: `Rashal10/baremetal-llm`, branch: `main`.
6. Set **App file** to `space/app.py` (or copy `demos/app.py` to Space root if HF only allows root `app.py`).
7. Add `space/requirements.txt` as the Space requirements file.

If HF expects files at repo root for a linked Space, duplicate:

```text
app.py              <- copy from space/app.py
README.md           <- copy from space/README.md
requirements.txt    <- copy from space/requirements.txt
```

## Option B: upload manually

Upload these three files from `space/` to a new Gradio Space:

- `README.md`
- `app.py`
- `requirements.txt`

`requirements.txt` installs the library from GitHub:

```text
git+https://github.com/Rashal10/baremetal-llm.git
```

Push your latest code **before** creating the Space so pip can install it.

## After deploy

Live URL:

`https://huggingface.co/spaces/Rashal10/baremetal-llm-demo`

Add to resume, LinkedIn, and the main README.

## GitHub Pages (docs)

1. Push to `main`.
2. Repo **Settings → Pages → Build: GitHub Actions**.
3. Site: `https://rashal10.github.io/baremetal-llm/`

Docs workflow: `.github/workflows/docs.yml`
