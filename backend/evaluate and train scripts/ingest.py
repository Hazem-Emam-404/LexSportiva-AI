import os
import re
import shutil
import pickle
import argparse
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever

from utils import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_CHROMA_DIR,
    DEFAULT_DOCSTORE_DIR,
    DEFAULT_PARENT_CHUNK_SIZE,
    DEFAULT_PARENT_OVERLAP,
    DEFAULT_CHILD_CHUNK_SIZE,
    DEFAULT_CHILD_OVERLAP,
    chunking_embedding,
)

# ---------------------------------------------------------------------------
# Data Structures & File Configuration
# ---------------------------------------------------------------------------
class FileProperties:
    def __init__(
        self,
        id: str,
        file_path: str,
        name: str,
        target_pages: str,
        llama_file: Optional[Any] = None,
        llama_parsed_result: Optional[Any] = None
    ):
        self.id = id
        self.file_path = file_path
        self.name = name
        self.page_ranges = {"target_pages": target_pages}
        self.llama_file = llama_file
        self.llama_parsed_result = llama_parsed_result
        self.docs: List[Document] = []


DEFAULT_FILES_CONFIG: Dict[str, FileProperties] = {
    "football": FileProperties(id="file_1", file_path="./files/Football.pdf", name="Football", target_pages="6-107"),
    "basketball": FileProperties(id="file_2", file_path="./files/Basketball.pdf", name="Basketball", target_pages="5-99"),
    "handball": FileProperties(id="file_3", file_path="./files/Handball.pdf", name="Handball", target_pages="5-81"),
    "tennis": FileProperties(id="file_4", file_path="./files/Tennis.pdf", name="Tennis", target_pages="5-44"),
    "boxing": FileProperties(id="file_5", file_path="./files/Boxing.pdf", name="Boxing", target_pages="5-49")
}

DEFAULT_PKL_PATH = "files_data.pkl"


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------
def basic_clean_text(text: str) -> str:
    """
    Cleans raw markdown text:
    1. Normalizes horizontal spaces and tabs.
    2. Normalizes 3+ newlines to exactly 2 (preserves paragraph breaks).
    3. Trims whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Robust Unpickler for cached files_data.pkl
# ---------------------------------------------------------------------------
class _FilePropertiesUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "FileProperties":
            return FileProperties
        return super().find_class(module, name)


def load_parsed_files(pkl_path: str = DEFAULT_PKL_PATH) -> Dict[str, FileProperties]:
    """Loads pre-parsed document dictionary from pickle file."""
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"Parsed file cache '{pkl_path}' not found. Run with --reparse to parse PDFs via LlamaCloud."
        )
    with open(pkl_path, "rb") as f:
        loaded_files = _FilePropertiesUnpickler(f).load()
    return loaded_files


# ---------------------------------------------------------------------------
# LlamaCloud Parsing Phase
# ---------------------------------------------------------------------------
def parse_with_llama_cloud(
    files_config: Dict[str, FileProperties] = DEFAULT_FILES_CONFIG,
    output_pkl: str = DEFAULT_PKL_PATH
) -> Dict[str, FileProperties]:
    """
    Parses all PDFs using LlamaCloud agentic parser with custom technical rules prompt,
    extracts pages into LangChain Document objects, and saves to pickle.
    """
    from llama_cloud import LlamaCloud

    api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise ValueError("LLAMA_CLOUD_API_KEY is not set in environment or .env file.")

    client = LlamaCloud()
    print("Connecting to LlamaCloud for document parsing...")

    # 1. Upload files
    for key, prop in files_config.items():
        print(f"Uploading {prop.name} ({prop.file_path})...")
        file_obj = client.files.create(file=prop.file_path, purpose="parse")
        prop.llama_file = file_obj

    # 2. Parse files with agentic prompt
    custom_prompt = """
    You are an expert technical extractor processing official sports rulebooks for a Retrieval-Augmented Generation (RAG) system. Extract all main document text accurately.
    
    When you encounter an image, apply the following strict filtering rules:
    
    1. IGNORE DECORATIVE IMAGES: Completely skip and do not describe any logos (e.g., federation crests), cover pages, atmospheric background graphics, or photographs of players in action. If an image contains no technical or factual rule data, omit it entirely.
    
    2. DESCRIBE FACTUAL/TECHNICAL IMAGES: If an image contains factual data—specifically court/pitch dimensions, equipment specifications, technical diagrams, or referee hand signals—you must transcribe and describe it in exhaustive detail.
    
    3. DESCRIPTION FORMAT: For factual images, provide a structured text representation. 
        - For court diagrams: List all specific measurements, lines, zones, and their relative distances.
        - For referee signals: Describe the exact physical posture, arm/hand placement, and movement required for the signal, as if giving instructions to a blind person. 
        - Explicitly state all numbers, labels, text callouts, and relevant colors present in the technical diagram.
    """

    for key, prop in files_config.items():
        print(f"Parsing {prop.name} pages {prop.page_ranges['target_pages']} via LlamaCloud...")
        result = client.parsing.parse(
            file_id=prop.llama_file.id,
            tier="agentic",
            version="latest",
            expand=["markdown", "metadata"],
            agentic_options={"custom_prompt": custom_prompt},
            page_ranges=prop.page_ranges,
            output_options={
                "markdown": {
                    "tables": {"merge_continued_tables": True, "output_tables_as_markdown": False}
                }
            }
        )
        prop.llama_parsed_result = result

        # 3. Create LangChain Documents
        docs = []
        for md_page, meta_page in zip(result.markdown.pages, result.metadata.pages):
            page_metadata = {
                "page_number": meta_page.page_number,
                "file_name": prop.name,
                "file_path": prop.file_path,
                "confidence": getattr(meta_page, "confidence", None)
            }
            doc = Document(
                page_content=basic_clean_text(md_page.markdown),
                metadata=page_metadata
            )
            docs.append(doc)

        prop.docs = docs
        print(f"Created {len(docs)} LangChain Documents for {prop.name}.")

    # Save to pickle
    with open(output_pkl, "wb") as f:
        pickle.dump(files_config, f)
    print(f"All parsed documents successfully saved to '{output_pkl}'.")
    return files_config


# ---------------------------------------------------------------------------
# Storage Cleanup (Wipe Old Chroma & DocStore)
# ---------------------------------------------------------------------------
def clean_generated_storage(
    chroma_dir: str = DEFAULT_CHROMA_DIR,
    docstore_dir: str = DEFAULT_DOCSTORE_DIR
):
    """
    Deletes the Chroma vectorstore and docstore directories so that chunking,
    splitters, and embeddings can be rebuilt cleanly from scratch.
    """
    print("\n--- Cleaning Existing Storage Directories ---")
    for directory in [chroma_dir, docstore_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory, ignore_errors=True)
            print(f"Removed: {directory}")
        else:
            print(f"Not found (clean): {directory}")
    print("Storage cleaned.")


# ---------------------------------------------------------------------------
# Chunking & Embedding Pipeline
# ---------------------------------------------------------------------------
def chunk_and_embed(
    loaded_files: Dict[str, FileProperties],
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    parent_chunk_size: int = DEFAULT_PARENT_CHUNK_SIZE,
    parent_overlap: int = DEFAULT_PARENT_OVERLAP,
    child_chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
    child_overlap: int = DEFAULT_CHILD_OVERLAP,
    chroma_dir: str = DEFAULT_CHROMA_DIR,
    docstore_dir: str = DEFAULT_DOCSTORE_DIR
) -> Tuple[ParentDocumentRetriever, Chroma, Any]:
    """
    Delegates to utils.chunking_embedding with custom splitters and is_for_embedding=True.
    """
    print(f"\n--- Chunking & Embedding with {embedding_model} ---")
    print(f"Parent Splitter: chunk={parent_chunk_size}, overlap={parent_overlap}")
    print(f"Child Splitter:  chunk={child_chunk_size}, overlap={child_overlap}")

    retriever, vectorstore, store = chunking_embedding(
        embedding_model=embedding_model,
        loaded_files=loaded_files,
        is_for_embedding=True,
        parent_chunk_size=parent_chunk_size,
        parent_overlap=parent_overlap,
        child_chunk_size=child_chunk_size,
        child_overlap=child_overlap,
        chroma_dir=chroma_dir,
        docstore_dir=docstore_dir
    )

    child_count = vectorstore._collection.count()
    parent_count = len(list(store.yield_keys()))

    print("\n=======================================================")
    print(" INGESTION & EMBEDDING COMPLETED SUCCESSFULLY")
    print("=======================================================")
    print(f"Parent Chunks in DocStore: {parent_count}")
    print(f"Child Chunks in Chroma VectorStore: {child_count}")
    print(f"Chroma Persist Directory: {chroma_dir}")
    print(f"DocStore Directory: {docstore_dir}")
    print("=======================================================\n")

    return retriever, vectorstore, store


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def run_pipeline(
    reparse: bool = False,
    clean: bool = True,
    parent_chunk_size: int = DEFAULT_PARENT_CHUNK_SIZE,
    parent_overlap: int = DEFAULT_PARENT_OVERLAP,
    child_chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
    child_overlap: int = DEFAULT_CHILD_OVERLAP,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    pkl_path: str = DEFAULT_PKL_PATH,
    chroma_dir: str = DEFAULT_CHROMA_DIR,
    docstore_dir: str = DEFAULT_DOCSTORE_DIR
):
    """
    Full pipeline execution:
    1. Parsing: re-parse via LlamaCloud if requested, otherwise load cached pkl.
    2. Clean: wipe existing Chroma and Docstore directories if requested.
    3. Chunk & Embed: rebuild Parent-Child index with updated parameters.
    """
    # 1. Parsing
    if reparse or not os.path.exists(pkl_path):
        print(f"Running LlamaCloud parsing (reparse={reparse}, file_exists={os.path.exists(pkl_path)})...")
        loaded_files = parse_with_llama_cloud(output_pkl=pkl_path)
    else:
        print(f"Loading cached parsed documents from '{pkl_path}'...")
        loaded_files = load_parsed_files(pkl_path=pkl_path)
        for k, prop in loaded_files.items():
            print(f" - {prop.name}: {len(prop.docs)} pages")

    # 2. Cleanup existing generated databases
    if clean:
        clean_generated_storage(chroma_dir=chroma_dir, docstore_dir=docstore_dir)

    # 3. Chunk & Embed from scratch
    retriever, vectorstore, store = chunk_and_embed(
        loaded_files=loaded_files,
        embedding_model=embedding_model,
        parent_chunk_size=parent_chunk_size,
        parent_overlap=parent_overlap,
        child_chunk_size=child_chunk_size,
        child_overlap=child_overlap,
        chroma_dir=chroma_dir,
        docstore_dir=docstore_dir
    )

    return retriever, vectorstore, store


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingestion, Chunking & Embedding Pipeline for Sports Rulebooks RAG"
    )
    parser.add_argument(
        "--reparse",
        action="store_true",
        help="Force re-parsing PDFs via LlamaCloud (overwrites files_data.pkl). Without this flag, cached parsed docs are used."
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do NOT delete existing Chroma and Docstore directories before running (default is to clean and regenerate from scratch)."
    )
    parser.add_argument(
        "--parent-chunk-size",
        type=int,
        default=DEFAULT_PARENT_CHUNK_SIZE,
        help=f"Chunk size for parent splitter (default: {DEFAULT_PARENT_CHUNK_SIZE})"
    )
    parser.add_argument(
        "--parent-overlap",
        type=int,
        default=DEFAULT_PARENT_OVERLAP,
        help=f"Chunk overlap for parent splitter (default: {DEFAULT_PARENT_OVERLAP})"
    )
    parser.add_argument(
        "--child-chunk-size",
        type=int,
        default=DEFAULT_CHILD_CHUNK_SIZE,
        help=f"Chunk size for child splitter (default: {DEFAULT_CHILD_CHUNK_SIZE})"
    )
    parser.add_argument(
        "--child-overlap",
        type=int,
        default=DEFAULT_CHILD_OVERLAP,
        help=f"Chunk overlap for child splitter (default: {DEFAULT_CHILD_OVERLAP})"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"HuggingFace embedding model (default: {DEFAULT_EMBEDDING_MODEL})"
    )

    args = parser.parse_args()

    run_pipeline(
        reparse=args.reparse,
        clean=not args.no_clean,
        parent_chunk_size=args.parent_chunk_size,
        parent_overlap=args.parent_overlap,
        child_chunk_size=args.child_chunk_size,
        child_overlap=args.child_overlap,
        embedding_model=args.embedding_model
    )
