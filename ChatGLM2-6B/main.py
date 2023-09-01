import sys

from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.llms.base import LLM
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModel, AutoConfig


class GLM(LLM):
    max_token: int = 2048
    temperature: float = 0.8
    top_p = 0.9
    tokenizer: object = None
    model: object = None
    history_len: int = 1024

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "GLM"

    def load_model(self, llm_device="gpu"):
        # model_config = AutoConfig.from_pretrained(model_name_or_path,
        #                                           trust_remote_code=True,
        #                                           cache_dir='ChatGLM2-6B/chatglm2-6b')
        self.tokenizer = AutoTokenizer.from_pretrained("chatglm2-6b", trust_remote_code=True, cache_dir="ChatGLM2-6B/chatglm2-6b")
        self.model = AutoModel.from_pretrained("chatglm2-6b", trust_remote_code=True, cache_dir="ChatGLM2-6B/chatglm2-6b").cuda()

    def _call(self, prompt: str, history: List[str] = [], stop: Optional[List[str]] = None):
        response, _ = self.model.chat(
            self.tokenizer, prompt,
            history=history[-self.history_len:] if self.history_len > 0 else [],
            max_length=self.max_token, temperature=self.temperature,
            top_p=self.top_p)
        return response


# model_path = "ChatGLM2-6B/chatglm2-6b"
# sys.path.append(model_path)
# tokenizer = AutoTokenizer.from_pretrained("chatglm2-6b", trust_remote_code=True, cache_dir=model_path)
# model = AutoModel.from_pretrained("chatglm2-6b", trust_remote_code=True, cache_dir=model_path).cuda()
if __name__ == '__main__':

    model = GLM()
    model.load_model()

    with open("../1.txt", errors='ignore') as f:
        content = f.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(content)
    docs = [Document(page_content=t) for t in texts]
    prompt_template = """对下面的文字做精简的摘要:
    
        {text}
    
        """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(model, chain_type="map_reduce", return_intermediate_steps=True, map_prompt=PROMPT, combine_prompt=PROMPT)
    summ = chain({"input_documents": docs}, return_only_outputs=True)
    print(summ['output_text'])