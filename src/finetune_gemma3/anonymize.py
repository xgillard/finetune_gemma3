"""Anonymize dataset."""
import asyncio
import logging

import pandas as pd
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

logging.getLogger("transformers").setLevel(logging.WARNING)

modelname  = "meta-llama/Llama-3.2-3B-Instruct"
dataset    = "./resources/mails_dataset_trim.csv"
max_length = 1000


df = pd.read_csv(dataset)  # noqa: PD901
df = df[ (df["request"].str.len() <= max_length) & (df["response"].str.len() <= max_length) ]  # noqa: PD901

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
model = AutoModelForCausalLM.from_pretrained(
    modelname,
    quantization_config = bnb_config,
    device_map="auto",
    attn_implementation="eager",
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1000,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=False,
    temperature=None,
    top_p=None,
    top_k=None,
)
llm  = HuggingFacePipeline(pipeline=pipe, model_id=modelname)
chat = ChatHuggingFace(llm=llm)

### VECTOR DB #################################################################
__DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
__DIGITA = "/home/xgillard/REPO/rag/agr/notebooks/digita"

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
ANON_RQ = ChatPromptTemplate.from_messages([
    ("system", """
     Anonymize the following email.
     Replace persons first names by Alice, Bob, Charlie, Dylan etc...
     Replace persons names by FOO, BAR, BAZ, etc...
     Keep the email adresses consistent with the replacement you've made.
     Change mail address and/or phone whenever needed.
     Do not change the format of the email.
     Do not change city names.
     Output anonymized email ONLY."""),
     ("human", "{input}"),
])
ANON_RSP = ChatPromptTemplate.from_messages([
    ("system", """
     Anonymize the following email.
     Replace persons first names by Alice, Bob, Charlie, Dylan etc...
     Replace persons names by FOO, BAR, BAZ, etc...
     Keep the email adresses consistent with the replacement you've made.
     Change mail address and/or phone whenever needed.
     Do not change the format of the email.
     Do not change city names.
     Output anonymized email ONLY."""),
     ("human", "{request}"),
     ("ai", "{anon}"),
     ("system", """
      Now do the same for the response to that email.
      Be consistent with how you anonymized the 1st email.
      """),
    ("human", "{response}"),
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

### CHAINS ###################################################################
def response(x:str) -> str:
    """Return the model response."""
    start = "<|start_header_id|>assistant<|end_header_id|>"
    ix = x.rfind(start)
    return x[ix+len(start):].strip() if ix != -1 else x


translate = TRANSLATE | chat | StrOutputParser() | response
anonymize = ANON_RQ | chat | StrOutputParser() | response

async def anon(rq: str, rsp: str) -> tuple[str, str, str]:
    """Anonymize both a request and a response."""
    trq = translate.ainvoke(rq)
    trsp= translate.ainvoke(rsp)

    trq = await trq
    arq = anonymize.ainvoke(trq)
    flow= (ROUTE_TO_FLOW | chat | StrOutputParser() | response).ainvoke(arq)
    trsp= await trsp
    arq = await arq
    arsp= (ANON_RSP | chat | StrOutputParser() | response).ainvoke({
        "request": trq,
        "anon": arq,
        "response": trsp,
    })

    flow = await flow
    ctx  = None
    match f:=flow.lower():
        case _ if "genealogy" in f:
            src = GENEALOGY.as_retriever(search_type="mmr", search_kwargs={"k":2})
            prt = FIND_RELEVANT.format(input=arq)
            ctx = await src.ainvoke(prt)
            ctx = "\n\n".join([c.page_content for c in ctx])
        case _ if "records" in f:
            src = RECORDS.as_retriever(search_type="mmr", search_kwargs={"k":2})
            prt = FIND_RELEVANT.format(input=arq)
            ctx = await src.ainvoke(prt)
            ctx = "\n\n".join([c.page_content for c in ctx])
        case _: # faq
            src = FAQ.as_retriever(search_type="mmr", search_kwargs={"k":2})
            prt = FIND_SIMILAR.format(input=arq)
            ctx = await src.ainvoke(prt)
            ctx = "\n\n".join([c.metadata["fulltext"] for c in ctx])

    arsp = await arsp
    return (arq, arsp, ctx)


async def main():
    results = []
    for _, row in df.iterrows():
        point = await anon(row["request"], row["response"])
        results.append(point)
        if len(results) % 10 == 0:
            # checkpoint
            out = pd.DataFrame.from_records(results, columns=["request", "response", "context"])
            out.to_csv("anonymized.csv")
    # save final dataset
    out = pd.DataFrame.from_records(results, columns=["request", "response", "context"])
    out.to_csv("anonymized.csv")

if __name__ == "__main__":
    asyncio.run(main())
