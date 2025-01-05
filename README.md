# IRS: The Intelligent Receipt Splitter

Tired of overpaying just because splitting the bill equally was easier? Now, pay only your fair share with this app! Snap a photo of your receipt, specify who had what, and let the app handle the rest. Handle tips, taxes, and if you're feeling a little charitable, credit card cashback as well!

Know Python? Then read on to run it locally.

Don't know enough Python? Watch this space.

## Screenshots

<p float="left">
<img src="/assets/images/Screenshot_1.jpg" width=15%>
<img src="/assets/images/Screenshot_2.jpg" width=15%>
<img src="/assets/images/Screenshot_3.jpg" width=15%>
</p>

## Setup environment

### Prerequisites

You will need [`ollama`](https://ollama.com/) installed and running with a model of your choice available. The default is `qwen2.5:7b`, while this is easily configurable.

### Python virtual environment

Use `uv` from the folks over at [astral.sh](https://github.com/astral-sh/uv). After cloning the repository, do:

```bash
uv sync --extra dev
source .venv/bin/activate # On MacOS/Linux, venv\Scripts\activate on Windows
export PYTHONPATH=$(pwd)
```

If you don't have `uv` or don't care for it, then create a virtual environment and install the package itself. Below is an example using `pyvenv`.

```bash
python3 -m venv venv # Python 3.6+
source .venv/bin/activate # On MacOS/Linux, venv\Scripts\activate on Windows
pip install .
export PYTHONPATH=$(pwd)
```

## Start the Gradio app

If everything's setup properly in the virtual environment, run:

```bash
python src/app/gradio_ui.py
```

By default, it should run at `0.0.0.0:7860`. If you want to run this on a mobile device while processing on your local machine, you will need to identify the IP address of your machine and be connected to the same network the machine is on. Then, in your mobile browser, navigate to:

```commandline
http://<machine-ip-address>:7860/
```

# Other stuff

## How it works

Creating an LLM app that is actually useful is rather hard - LLMs are notoriously overconfident when wrong. Not to mention, a major provider is likely to store the conversations you have with it. This is an attempt to work around these limitations by:

1. **Performing Optical Character Recognition (OCR) on images to extract text.** OCR frameworks aren't generative models, so they are far less likely to go wrong. This avoids the unreliability of LLMs.
2. **Using LLMs as an Intelligent Document Processing (IDP) layer to extract relevant fields from the OCR'd receipt text.** LLMs are great for this - getting the right context around text the same way a human would. This is also done entirely locally, by using `ollama` and downloaded weights (the default is a 7B model, which can run on machines with as little as 16GB RAM).
3. **Incorporating a Human-in-the-Loop workflow to verify uncertain data extractions.** A human - you, will be provided a UI to correct any unreliable data the LLM may have extracted in Step (2), and configure exactly how you want to split the receipt.

Hope you find this useful!

## Thanks

This web-app uses the following components aside from Gradio:

1. [`surya`](https://github.com/VikParuchuri/surya) for OCR capabilities.
2. [`ollama`](https://github.com/ollama/ollama) for running LLMs locally.
3. All LLMs providers that open sourced their weights.

## FAQ

> Did you know that the abbreviation IRS conflicts with the Internal Revenue Service?

Yes.
