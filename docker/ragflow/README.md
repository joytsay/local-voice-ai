# RAGFlow integration

The main Compose file remotely includes the official RAGFlow v0.26.4 service
definitions from `https://github.com/infiniflow/ragflow`. RAGFlow runs in its
own `ivr-ragflow` container; it is not installed in the Gradio or LLM container.
The local Compose overlay builds that container with `docker/ragflow.Dockerfile`.
The Dockerfile clones the pinned RAGFlow source and builds it for the host
architecture, avoiding the upstream AMD64-only image on an AGX Orin. The
official include retains RAGFlow's database, object storage, search engine, and
cache service definitions.

## Setup

1. Build and start the ARM64 RAGFlow container and its services:

   ```sh
   docker compose \
     -f docker-compose.yml \
     -f docker-compose.local.yml \
     --profile cpu \
     --profile elasticsearch \
     up -d --build ragflow-cpu es01 mysql minio redis
   ```

   The first native build is large and can take a long time. RAGFlow's UI is
   published at `http://localhost` and its REST API at
   `http://localhost:9380`.
2. In RAGFlow, configure an embedding model, create a dataset using the General
   chunk method, and create an API key.
3. Put the key and dataset ID in the root `.env` as `RAGFLOW_API_KEY` and
   `RAGFLOW_DATASET_IDS`.
4. Run the importer inside the Gradio container:

   ```sh
   docker compose exec gradio python /app/docker/ragflow/import_wiki.py
   ```

5. Wait until all documents show `DONE` in RAGFlow before using the voice flow.

When the API key and dataset ID are blank, the Gradio app loads all local wiki
pages as a development fallback. Once both are configured, retrieval errors are
reported instead of silently bypassing RAGFlow.

The runtime uses RAGFlow's `POST /api/v1/retrieval` endpoint. Retrieved chunks
are appended to `knowledge/system-prompt.md`; the existing local LLM remains
responsible for producing the normalized transcript.
