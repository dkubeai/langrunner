from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

llamaindex_required = ['llama-index==0.10.53', 'llama-index-core==0.10.53.post1', 'llama-index-embeddings-adapter==0.1.3', 'llama-index-embeddings-openai==0.1.10',
        'llama-index-finetuning==0.1.10', 'llama-index-llms-huggingface==0.2.4', 'llama-index-llms-openai==0.1.25', 'llama-index-readers-file==0.1.29', 'mistralai==0.4.2']

langchain_required = ['langchain==0.2.10', 'langchain-community==0.2.9', 'langchain-core==0.2.22', 'langchain-huggingface==0.0.3', 'langchain-openai==0.1.17', 'langchain-text-splitters==0.2.2', 'langchainhub==0.1.20']

all_required = llamaindex_required + langchain_required

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="langrunner",
    version="0.1",
    description="remote execution layer for lang resources.",
    author="ahmed",
    long_description=long_description,
    long_description_content_type="text/markdown",    
    author_email="ahmed@dkube.io",
    url="https://github.com/mahmedk/langrunner.git",
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
    python_requires=">=3.10",
    extras_require={
        'llama_index': llamaindex_required,
        'langchain': langchain_required,
        'all': all_required
    }
)
