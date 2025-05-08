"""Anonymize dataset."""
import logging
import sys

import torch
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
)
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)
from peft import (
    AutoPeftModelForCausalLM,
)
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

modelname   = "meta-llama/Llama-3.2-1B-Instruct"
adaptername = "xaviergillard/Llama-3.2-1B-Instruct-arkey_emails-qlora"

max_concurr = 10
__DIGITA    = "./resources/digita"
__DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

### LOGGING ###################################################################
log = logging.getLogger("eval")
log.addHandler(logging.StreamHandler(sys.stdout))
log.setLevel(logging.INFO)

### MODEL #####################################################################
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    modelname,
)
model = AutoPeftModelForCausalLM.from_pretrained(
    adaptername,
    quantization_config = bnb_config,
    attn_implementation="eager",
)
model = model.merge_and_unload()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1000,
    pad_token_id=tokenizer.eos_token_id,
    temperature=None,
    top_p=None,
    top_k=None,
    #device=__DEVICE,
)
llm  = HuggingFacePipeline(pipeline=pipe, model_id=modelname)
chat = ChatHuggingFace(llm=llm)

### VECTOR DB #################################################################
# The FAQ database
FAQ = Chroma(
    "faq",
    HuggingFaceEmbeddings(
        model_name   = "intfloat/multilingual-e5-large-instruct",
        model_kwargs = {"device": __DEVICE, "model_kwargs": {"torch_dtype": "float16"}},
    ),
    persist_directory=__DIGITA,
)

# The Getting started with Genealogy database
GENEALOGY = Chroma(
    "genealogy",
    HuggingFaceEmbeddings(
        model_name   = "intfloat/multilingual-e5-large-instruct",
        model_kwargs = {"device": __DEVICE, "model_kwargs": {"torch_dtype": "float16"}},
    ),
    persist_directory=__DIGITA,
)

# The Getting started with Genealogy database
RECORDS = Chroma(
    "records",
    HuggingFaceEmbeddings(
        model_name   = "intfloat/multilingual-e5-large-instruct",
        model_kwargs = {"device": __DEVICE, "model_kwargs": {"torch_dtype": "float16"}},
    ),
    persist_directory=__DIGITA,
)
### PROMPTS ###################################################################
TRANSLATE = ChatPromptTemplate.from_messages([
    ("system", "Translate this email to English. Output translation ONLY."),
    ("human", "{input}"),
])

ROUTE_TO_FLOW = ChatPromptTemplate.from_template("""
You are an AI assistant whose role it is to route requests adressed to the
support of the Belgian State Archive website so as to make sure it is handled
by the most knowledgeable of your colleagues.

Given the request below, reply one of the following literals:
* "faq" if you think the question is general enough and can be answered by reading the website FAQ.
* "genealogy" when the user needs help to start a building a family tree or finding a specific person.
* "records" when the user needs practical guidance on how to conduct a search through analysis of records.

# User Request:
{input}
""")

FIND_RELEVANT = PromptTemplate.from_template("""
Instruct: Retrieve the items that are most relevant to answering to the user's query.
Query: {input}
""")

FIND_SIMILAR = PromptTemplate.from_template("""
Instruct: Retrieve the items that are most similar to the user's query.
Query: {input}
""")

RESPOND = ChatPromptTemplate.from_messages([
        ("system", "Respond to this user email using the given pieces of information only."),
        ("human", "{email} \n\n\n# Information you can use:\n{context}"),
    ])

### CHAINS ###################################################################
def response(x:str) -> str:
    """Return the model response."""
    start = "<|start_header_id|>assistant<|end_header_id|>"
    ix = x.rfind(start)
    return x[ix+len(start):].strip() if ix != -1 else x

translate = TRANSLATE | chat | StrOutputParser() | response


def respond(email: str) -> str:
    """Anonymize both a request and a response."""
    trq  = translate.invoke(email)
    flow = (ROUTE_TO_FLOW | chat | StrOutputParser() | response).invoke(trq)
    ctx  = None
    match f:=flow.lower():
        case _ if "genealogy" in f:
            src = GENEALOGY.as_retriever(search_type="mmr", search_kwargs={"k":2})
            prt = FIND_RELEVANT.format(input=trq)
            ctx = src.invoke(prt)
            ctx = "\n\n".join([c.page_content for c in ctx])
        case _ if "records" in f:
            src = RECORDS.as_retriever(search_type="mmr", search_kwargs={"k":2})
            prt = FIND_RELEVANT.format(input=trq)
            ctx = src.invoke(prt)
            ctx = "\n\n".join([c.page_content for c in ctx])
        case _: # faq
            src = FAQ.as_retriever(search_type="mmr", search_kwargs={"k":2})
            prt = FIND_SIMILAR.format(input=trq)
            ctx = src.invoke(prt)
            ctx = "\n\n".join([c.metadata["fulltext"] for c in ctx])

    return (RESPOND | chat | StrOutputParser() | response).invoke({"email": trq, "context": ctx})


def main() -> None:
    """Run the main entry point."""
    email = """
    Bonjour et Meilleurs Voeux
    Je recherche dans vos archives numérisés de Bruxelles, l'acte de mariage 2117 du 30/11/1901
    entre MACHIN Jean Eugène & BAZAER Anna Alexandrine.
    Je ne trouve pas dernier trimestre 1901.
    Bien à vous
    """
    print(respond(email))

if __name__ == "__main__":
    main()
