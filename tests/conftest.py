def pytest_addoption(parser):
    parser.addoption(
        "--ollama-model-name",
        action="store",
        default="tulu3:8b",
        help="Specify the ollama model name to use. It should exist on the system you are running this from.",
    )
