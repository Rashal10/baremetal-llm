# Interactive Demos

Here are the ways you can run and play with the interactive models and visualizations.

## Google Colab (Instant, Zero Setup)

The easiest way to see the model run without cloning or downloading anything is through our pre-built Google Colab notebooks. It is completely free and executes in any browser.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rashal10/baremetal-llm/blob/main/notebooks/01_quick_demo.ipynb)

Inside the notebook you can:
- Automatically train a tiny character-level model
- Sample text completions dynamically
- Play with hyperparameters like context length and embed dimensions

## Local Gradio UI

If you want the full graphical experience with the visualizers (including attention maps and mixture-of-expert routing histograms), you can run the Gradio interface locally.

```bash
# Install the package with demo dependencies
pip install -e ".[demo]"

# Launch the Gradio server
python demos/app.py
```

Open **http://127.0.0.1:7860** in your browser.

### Shareable Public Links

If you are actively interviewing or collaborating with someone and want to give them access to your running model, you can launch the app with the `--share` flag:

```bash
python demos/app.py --share
```

This will automatically spin up a secure, temporary public URL (for example: `https://abcd1234.gradio.live`). This link will remain active for 72 hours and runs the model directly on your machine's hardware for free.

## GitHub Pages Documentation

This documentation site is built and deployed automatically whenever changes are pushed to the main branch of our repository.

- **Docs Site URL:** [rashal10.github.io/baremetal-llm](https://rashal10.github.io/baremetal-llm/)
- **Configuration:** Managed via `.github/workflows/docs.yml` and the standard `gh-pages` branch.
