def pytest_addoption(parser):
    parser.addoption(
        "--ollama-model-name",
        action="store",
        default="qwen2.5:7b",
        help="Specify the ollama model name to use. It should exist on the system you are running this from.",
    )
