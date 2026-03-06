"""
    This module convert the PDF files from any dir
    into a semantic vectorial DB in the output location in config.py
"""

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import init_embeddings
from langchain_chroma import Chroma
from src.config import EMBEDDING_MODEL, CHROMA_DB_PATH


def load_pdfs(pdf_dir: str):
    """
    Loads a pdf directory and return as a list of docume
    """
    loader = PyPDFDirectoryLoader(pdf_dir)
    documents = loader.load()
    print(f"[SUCCESS] - {len(documents)} loaded")

    return documents


def split_documents(documents):
    """
    Recive a list of documents from PytPDFDirectoryLoader
    and returns the documents splitted in batches
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", " "]
    )

    return splitter.split_documents(documents)


def generate_chunks_ids(chunks):
    """
    Chunk es cada fragmento generado por el splitter, con su
    metadata correspondiente, los vamos a unificar con un ID que indique a qué
    contrato y página pertenecen, para luego poder identificarlo fácilmente en
    la base de datos vectorial
    """
    last_source = None
    last_page = None
    chunk_index = 0

    # Enumera los chunks que pertenecen a la misma pagina
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknow")
        page = chunk.metadata.get("page", 0)

        # Si el fragmento pertenece a la misma fuente y página
        # que el fragmento anterior, incrementa el índice del fragmento
        if source == last_source and page == last_page:
            chunk_index += 1
        else:
            # Si el fragmento pertenece a una fuente o página diferente,
            # reinicia el índice del fragmento
            chunk_index = 0

        chunk.metadata["id"] = f"{source}:{page}:{chunk_index}"
        # Almacenamos esta ultima fuente y pagina
        last_source = source
        last_page = page

    return chunks


def generate_indexes(pdf_dir: str):
    """
    Generate the embeddings just giving the path location
    """
    embeddings = init_embeddings(model=EMBEDDING_MODEL)

    # Conectar a la VDB
    vdb = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )

    # Cargar, splitear y generar los IDs
    pdfs = load_pdfs(pdf_dir)
    splited_pdfs = split_documents(pdfs)
    indexed_pdf_parts = generate_chunks_ids(splited_pdfs)

    # Obtenemos los ids nuevos a insertar
    candidate_ids = [c.metadata["id"] for c in indexed_pdf_parts]
    existing_ids = vdb.get(ids=candidate_ids)["ids"]

    # Evitar ids duplicados y solo agregar los nuevos a la vdb
    new_chunks = [c for c in indexed_pdf_parts
                  if c.metadata["id"] not in existing_ids
                  ]

    if new_chunks:
        print(f"[INFO] - Agregando {len(new_chunks)} nuevos chunks")
        try:
            vdb.add_documents(
                documents=new_chunks,
                # Redunda poner el "id" pero es necesario para el llm
                ids=[c.metadata["id"] for c in new_chunks]
            )
            print(
                    f"[SUCCESS] - {len(new_chunks)} chunks indexados en "
                    f"{CHROMA_DB_PATH}"
                )
        except Exception as e:
            raise e
    else:
        print("[INFO] - No hay chunks nuevos para indexar")

    return vdb


if __name__ == "__main__":
    generate_indexes("./contracts")
