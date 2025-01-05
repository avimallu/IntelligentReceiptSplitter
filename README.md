# IRS: The Intelligent Receipt Splitter

Tired of overpaying just because splitting the bill equally was easier? Now, pay only your fair share with this app! Snap a photo of your receipt, specify who had what, and let the app handle the rest. Handle tips, taxes, and if you're feeling a little charitable, credit card cashback as well!

Know Python? Then read on to run it locally.

Don't know Python? Watch this space.

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

## Other stuff

> Did you know that the abbreviation IRS conflicts with the Internal Revenue Service?

Yes.
