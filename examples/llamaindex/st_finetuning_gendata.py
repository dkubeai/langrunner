import json

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode

TRAIN_FILES = ["./data/10k/lyft_2021.pdf"]
VAL_FILES = ["./data/10k/uber_2021.pdf"]

TRAIN_CORPUS_FPATH = "./data/train_corpus.json"
VAL_CORPUS_FPATH = "./data/val_corpus.json"

def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        print(f"Loaded {len(docs)} docs")

    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)
    nodes = nodes[:10]

    if verbose:
        print(f"Parsed {len(nodes)} nodes")

    return nodes


train_nodes = load_corpus(TRAIN_FILES, verbose=True)
val_nodes = load_corpus(VAL_FILES, verbose=True)

from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.llms.openai import OpenAI

train_dataset = generate_qa_embedding_pairs(train_nodes, llm=OpenAI(model="gpt-3.5-turbo"))
val_dataset = generate_qa_embedding_pairs(val_nodes, llm=OpenAI(model="gpt-3.5-turbo"))

train_dataset.save_json("train_dataset.json")
val_dataset.save_json("val_dataset.json")
