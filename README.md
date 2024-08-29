![Langrunner](images/title.png)

> **LATEST RELEASE / DEVELOPMENT VERSION**: The [main](https://github.com/dkubeai/langrunner/tree/main) branch tracks the latest version: [0.1](https://github.com/dkubeai/langrunner/tree/v0.9.1.1). For the latest development version, checkout the [dev](https://github.com/dkubeai/langrunner/tree/dev) branch.

> **DISCLAIMER**: The project is currently under active development. During this time, you may encounter breaking changes in the API or implementation that could lead to unexpected behavior. We do not currently recommend using this project in production until the stable release is available.

We are working diligently toward releasing a stable version as soon as possible. Your feedback and contributions are greatly valued as we build a reliable LLM execution toolkit. Please note that the examples provided in the documentation are intended for educational purposes and initial exploration, not for production use.

## Overview

Langrunner is a tool designed to simplify the development of generative AI workflows by enabling remote execution of compute-intensive tasks. It integrates seamlessly with frameworks like Llamaindex and LangChain, allowing developers to offload specific code blocks to clusters with the appropriate resources (e.g., GPUs) without needing to containerize or manually manage deployment.

### Supported Environments

- **Public Cloud Providers**:  
  - Amazon Web Services (AWS)  
  - Google Cloud Platform (GCP)  
  - Microsoft Azure  

- **Private AI Infrastructure**:  
  - Kubernetes (any provider like rancher, gke, eks ..)

- **[FUTURE]  Orchestration Frameworks**:  
  - Slurm  
  - Ray
  - Kubeflow

### Key Features

- **Remote Execution**: Execute specific sections of your code on remote clusters, including AWS, GCP, Azure, or Kubernetes-based infrastructures.
- **Seamless Integration**: Directly integrates with your existing LangChain and Llamaindex workflows, requiring minimal changes to your codebase.
- **Automatic Artifact Management**: All artifacts generated by remote execution are automatically pulled back into your local environment, allowing for uninterrupted workflow continuation.
- **Resource Efficiency**: Schedule tasks on clusters with the exact resources needed, optimizing both time and cost.
- **Simple Configuration**: Run tasks anywhere with minimal configuration changes, providing flexibility across different computing environments.

## 🛠️ Built With

Langrunner is built on top of an amazing opensource project [SkyPilot](https://skypilot.readthedocs.io/) - A robust AI infrastructure project that powers scalable and efficient AI deployments.

We are deeply thankful to the SkyPilot team for their contributions to the AI community.

## How to setup for development

### 1. Create a Conda Environment

To start, create and activate a new Conda environment:

```bash
conda create -n lgrenv python=3.10.14 -y
conda activate lgrenv
```

### 2. Install Core Langrunner Dependencies

Navigate to the langrunner directory and install the core dependencies:

```bash
pip install .
```

### 3. Integration with LlamaIndex

Developing applications with Llamaindex or to run LlamaIndex examples, install the necessary packages:

```bash
pip install .[llama_index]
```

### 4. Integration with LangChain

Developing applications with Langchain or to run LangChain examples, install the necessary packages:

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

## 🛣️ Roadmap
- [ ] Llamaindex Integration
  - [x] Finetuning with langrunner
    - [x] Sentence transformers finetuning
    - [x] Implement Supervised Finetuning trainer and support it with langrunner
    - [x] Implement Flagembedding based finetuning and support it with langrunner
  - [ ] RAG Pipelines
    - [ ] Ingestion pipelines with remote schedule of document processing and embeddings generation.
    - [ ] Evaluation pipelines.
  - [ ] Llama packs
    - [ ] Remote runnables for llamapacks
- [ ] Langchain Integration
  - [ ] QnA Chain
    - [x] Servable for huggingface model used in the chain
  - [ ] Servable for many QnA chains in langchain
    - [ ] Support QnA over an API
  - [ ] Agents
    - [ ] Deploy and serve agents on an API
- [ ] Remote execution environments
  - [x] AWS
  - [x] GCP
  - [ ] AZURE
  - [x] KUBERNETES
  - [ ] SLURM
  - [ ] RAY

## 🤝 Contributing
We welcome contributions from the community! Please feel free to open issues or submit pull requests to help improve Langrunner.

The examples residing in this repository are great starting points. Please see the sections [How to setup for development] and [Running Examples] above.

- Use [Github Flow](https://docs.github.com/en/get-started/using-github/github-flow), all the code changes can be submitted via pull requests.
- Fork the repo and create your branch from `main`.
- Implement a feature or fix an issue
- Make sure your code lints.
- Create PR with branch name format of <issue_number>-<short_name>-

## Report a bug with Github [issues](https://github.com/dkubeai/langrunner/issues)

Discovered a bug ? Please raise an issue or reach out to us on [Slack] (https://slack.com/invite/dkubeai/langrunner)

Ready to help? Take on the issue and send us a Pull Request.

## New Feature requests

Please raise an issue and tag it as an `FeatureRequest`. Provide as much information as possible about the new feature, and we will schedule it for implementation in the project.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
