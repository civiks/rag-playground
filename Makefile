.DEFAULT_GOAL := help

help:           ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:        ## uv sync — install deps into .venv
	uv sync

ingest:         ## Parse corpus/ PDFs and build all Qdrant collections
	uv run python ingest.py

ingest-one:     ## Rebuild one strategy (e.g. make ingest-one STRATEGY=semantic)
	uv run python ingest.py --strategy $(STRATEGY)

app:            ## Run the Streamlit app on :8501
	uv run streamlit run app.py

phoenix:        ## Run the Phoenix trace UI on :6006
	uv run phoenix serve

eval:           ## Run the full eval matrix (4 configs, weakest → strongest)
	PYTHONPATH=. uv run python evals/run_evals.py

eval-one:       ## Run one config (e.g. make eval-one CHUNKING=docling_hybrid RETRIEVAL=hybrid)
	PYTHONPATH=. uv run python evals/eval.py \
		--chunking $(CHUNKING) --retrieval $(RETRIEVAL) \
		$(if $(RERANK),--rerank,) $(if $(REWRITE),--rewrite $(REWRITE),)

clean:          ## Wipe ingested index + caches (keeps corpus/)
	rm -rf data/qdrant data/docling_cache data/contextual_cache.json

clean-all: clean ## Also wipe eval outputs
	rm -rf data/eval

.PHONY: help install ingest ingest-one app phoenix eval eval-one clean clean-all
