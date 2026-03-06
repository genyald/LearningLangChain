# Configurations for the models
GENERATION_MODEL = "gpt-5-nano"
QUERY_MODEL = "gpt-5-nano"

# Config for the embeddings
EMBEDDING_MODEL = "openai:text-embedding-3-small"
CHROMA_DB_PATH = "./chroma_db"

# Config fo the retrivers
SEARCH_TYPE = "mmr"  # Most
MMR_DIVERSITY_LAMBDA = 0.7  # Balance in relevance (1 - most relevant)
MMR_FETCH_K = 20  # Initial docs to fetch before apply mmr
SEARCH_K = 2  # The elements numbers to keep
