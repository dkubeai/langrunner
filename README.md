## HOW TO INSTALL
#### Create conda environment

`conda create -n lgrenv python=3.10.14 -y`

`conda activate lgrenv`

#### Install core langrunner deps
`cd langrunner; pip install .` 

#### Install required llamaindex packages [required to run llamaindex examples]
`cd langrunner; pip install .[llama_index]`

#### Install required langchain packages [required to run langchain examples]
`cd langrunner; pip install .[langchain]`

## HOW TO RUN
#### Llamaindex sentence transformers finetuning
`cd examples/llama_index/sentence_transformers; python finetuning.py`

#### Llamaindex supervised finetuning
`cd examples/llama_index/sft; python finetuning.py`

#### Llamaindex flagembedding finetuning
`cd examples/llama_index/flagembedding; python finetuning.py`

#### Langchain Simple QnA chain with a huggingface model
`cd examples/langchain; python qna_chain.py`
