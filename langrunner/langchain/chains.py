from langchain_core.runnables import Runnable
from langchain_core.runnables.utils import Input,Output

from langchain_core.runnables.config import RunnableConfig

class RunnableRemote(Runnable):

    _server_deployed: bool = False

    def __init__(self, chain_obj):
        chain_obj.invoke = self.invoke


    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        #MAK - TODO - Deploy the chain to run remotely
