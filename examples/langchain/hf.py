from langchain_huggingface import HuggingFacePipeline


import langrunner
HuggingFacePipeline = langrunner.runnable(HuggingFacePipeline)
hf = HuggingFacePipeline.from_model_id(model_id="gpt2", task="text-generation")
print(hf.invoke("what are transformers?"))
