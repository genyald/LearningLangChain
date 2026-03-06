"""
This is the main file that controls all the flow of the agent
"""

# Default init model and embeddings
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers import MultiQueryRetriever

from langchain_chroma import Chroma

# Import the configurations
from src.config import (
    GENERATION_MODEL,
    QUERY_MODEL,
    EMBEDDING_MODEL,
    CHROMA_DB_PATH,
    SEARCH_TYPE,
    MMR_DIVERSITY_LAMBDA,
    MMR_FETCH_K,
    SEARCH_K,
)

# PROMPT TEMPLATES
from src.prompt import (
    RAG_TEMPLATE,
    MULTI_QUERY_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
    RELEVANCE_PROMPT,
)


# Initizlize the models and the embeddings
query_model = init_chat_model(model=QUERY_MODEL, model_provider="openai")
generation_model = init_chat_model(
    model=GENERATION_MODEL, configurable_fields="any", model_provider="openai"
)
embeddings = init_embeddings(model=EMBEDDING_MODEL, provider="openai")


def initialize_rag_system():
    """
    Initialize the Vectorial DB
    """
    vectors = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH
        )

    # Un retriever es un componente que se encarga de recuperar los
    # documentos relevantes de la base de datos vectorial en función de
    # la consulta del usuario. En este caso se utiliza un retriever
    # basado en el modelo de lenguaje para generar múltiples versiones
    # de la consulta del usuario y recuperar documentos relevantes
    # utilizando técnicas como MMR (Maximal Marginal Relevance) para
    # diversificar los resultados.

    base_retriver = vectors.as_retriever(  # Init the retriver
        search_type=SEARCH_TYPE,
        search_kwargs={
            "lambda_mult": MMR_DIVERSITY_LAMBDA,
            "fetch_k": MMR_FETCH_K,
            "k": SEARCH_K,
        },
    )

    # Custom prompt fot MultiQueryRetriver
    multiquery_prompt = PromptTemplate.from_template(template=MULTI_QUERY_PROMPT)

    # MultiQueryRetriver with custom prompt
    mmr_multi_retriver = MultiQueryRetriever.from_llm(
        retriever=base_retriver,
        llm=query_model,
        prompt=multiquery_prompt
    )

    prompt = PromptTemplate.from_template(RAG_TEMPLATE)

# response = query_model.invoke(
#   "Hola!",
#   # config={}
# )

# print(response.content)
