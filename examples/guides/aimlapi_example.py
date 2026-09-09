"""Example: Running Cognee on aimlapi.com.

aimlapi.com is an OpenAI-compatible gateway serving hundreds of chat models
behind one key. Cognee reaches it through litellm, so no adapter is needed:
set ``LLM_PROVIDER="custom"`` and prefix the catalogue id with ``aiml/``.

- LLM Provider: aimlapi.com (``aiml/openai/gpt-4o-mini``)
- Embeddings: aimlapi.com (``openai/text-embedding-3-large``)
- Local embedded database stack (Kuzu, LanceDB, SQLite)

Requires an aimlapi.com key exported as ``LLM_API_KEY``. Get one at
https://aimlapi.com/app/keys — the model catalogue is public at
https://api.aimlapi.com/v1/models?include=all
"""

import os
import asyncio
import tempfile
from pathlib import Path

# Setup temp directory to keep this example self-contained
_DATA_DIR = tempfile.mkdtemp(prefix="cognee_aimlapi_example_")
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
os.environ["CACHING"] = "false"

# Configure aimlapi.com environment settings.
# LLM_PROVIDER must be "custom": cognee's provider enum is deliberately small and
# an "aiml/" prefix alone raises ProviderNotDeducibleError, exactly like "openrouter/".
os.environ["LLM_PROVIDER"] = "custom"
os.environ["LLM_MODEL"] = "aiml/openai/gpt-4o-mini"
os.environ["LLM_ENDPOINT"] = "https://api.aimlapi.com/v1"
os.environ.setdefault("LLM_API_KEY", "")  # export your key before running

# Embeddings are configured independently of the LLM. litellm has no "aiml"
# embedding route, so the embedding model carries NO "aiml/" prefix -- it is the
# endpoint that points the OpenAI-compatible route at aimlapi.com.
os.environ["EMBEDDING_PROVIDER"] = "custom"
os.environ["EMBEDDING_MODEL"] = "openai/text-embedding-3-large"
os.environ["EMBEDDING_ENDPOINT"] = "https://api.aimlapi.com/v1"
os.environ["EMBEDDING_API_KEY"] = os.environ["LLM_API_KEY"]
os.environ["EMBEDDING_DIMENSIONS"] = "3072"
os.environ["EMBEDDING_MAX_COMPLETION_TOKENS"] = "8191"

import cognee  # noqa: E402
from cognee.modules.search.types import SearchType  # noqa: E402
from cognee.infrastructure.llm.config import get_llm_config  # noqa: E402

# Force local embedded stack configuration
cognee.config.set_graph_database_provider("kuzu")
cognee.config.set_vector_db_provider("lancedb")
cognee.config.data_root_directory(str(Path(_DATA_DIR) / "data"))
cognee.config.system_root_directory(str(Path(_DATA_DIR) / "system"))


SAMPLE_TEXT = """\
Cognee is an open-source library that helps developers turn documents into AI memory.
It builds semantic graphs, indexes entities, and stores vectors to enable structured retrieval.
Cognee supports hosted gateways such as aimlapi.com as well as local execution via Ollama.
"""


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


async def main() -> None:
    if not os.environ.get("LLM_API_KEY"):
        raise SystemExit("Export LLM_API_KEY with your aimlapi.com key before running.")

    # Start from a clean slate in isolated directory
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    banner("REMEMBER USING AIMLAPI.COM")
    llm_config = get_llm_config()
    print(f"Using LLM: {llm_config.llm_model}")
    print(f"Using Embeddings: {os.environ.get('EMBEDDING_MODEL')}")

    await cognee.remember(SAMPLE_TEXT, dataset_name="aimlapi_demo", self_improvement=False)
    print("Knowledge graph built successfully.")

    banner("RECALL")
    query = "What does Cognee help developers do?"
    results = await cognee.recall(
        query_text=query,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=["aimlapi_demo"],
    )
    print(f"Query: {query}")
    print("Recall Results:")
    print(results[0].text if results else "<no results>")


if __name__ == "__main__":
    asyncio.run(main())
