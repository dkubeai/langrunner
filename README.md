![Langrunner](images/title.png)

## Overview

The Langrunners Execution Plugin Framework provides a versatile solution for running LLM (Large Language Models) and LMM (Large Multimodal Models) application frameworks, such as LLamaIndex, LangChain, and Autogen, across a variety of environments without requiring code modifications.

### Supported Environments

- **Public Cloud Providers**:  
  - Amazon Web Services (AWS)  
  - Google Cloud Platform (GCP)  
  - Microsoft Azure  

  Including their respective Kubernetes environments:
  - AWS Elastic Kubernetes Service (EKS)  
  - Google Kubernetes Engine (GKE)  
  - Azure Kubernetes Service (AKS)  

- **Private AI Infrastructure Providers**:  
  - Nutanix  
  - VMWare  
  - NVIDIA  

- **Orchestration Frameworks**:  
  - Slurm  
  - Ray  

### Key Features

- **Infrastructure Independence**: Enables seamless execution of compute and memory-intensive tasks such as fine-tuning, data ingestion, and evaluation pipelines.
- **Simple Configuration**: Run tasks anywhere with minimal configuration changes, providing flexibility across different computing environments.


## Installation Guide

### 1. Create a Conda Environment

To start, create and activate a new Conda environment:

```bash
conda create -n lgrenv python=3.10.14 -y
conda activate lgrenv
```

### 2. Install Core Langrunner Dependencies

Navigate to the langrunner directory and install the core dependencies:

```bash
cd langrunner
pip install .
```

### 3. Install Required Packages for LlamaIndex

To run LlamaIndex examples, install the necessary packages:

```bash
pip install .[llama_index]
```

### 4. Install Required Packages for LangChain

To run LangChain examples, install the necessary packages:

```bash
pip install .[langchain]
```

## Running Examples

### LlamaIndex Examples

- Sentence Transformers Fine-Tuning
  
  Navigate to the sentence_transformers directory and run:

  ```bash
  cd examples/llama_index/sentence_transformers
  python finetuning.py
  ```

- Supervised Fine-Tuning
  
  Navigate to the sft directory and run:

  ```bash
  cd examples/llama_index/sft
  python finetuning.py
  ```

- FlagEmbedding Fine-Tuning
  
  Navigate to the flagembedding directory and run:

  ```bash
  cd examples/llama_index/flagembedding
  python finetuning.py
  ```

### Langchain Examples

- QnA chain with huggingface model
  
  Navigate to the examples/langchain directory and run:

  ```bash
  cd examples/langchain
  python qa_chain.py
  ```
  
