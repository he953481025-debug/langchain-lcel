import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAI

load_dotenv()

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
model = OpenAI(model="gpt-3.5-turbo-instruct")
output_parser = StrOutputParser()

chain = prompt | model | output_parser


def rag_search_example():
    # Requires:
    # pip install langchain docarray tiktoken

    from langchain_community.vectorstores import DocArrayInMemorySearch
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough
    from langchain_openai.chat_models import ChatOpenAI
    from langchain_openai.embeddings import OpenAIEmbeddings

    vectorstore = DocArrayInMemorySearch.from_texts(
        ["harrison worked at kensho", "bears like to eat honey"],
        embedding=OpenAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever()

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI()
    output_parser = StrOutputParser()
    retriever_result = retriever.invoke("where did harrison work?")
    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    # ressul = setup_and_retrieval.invoke("where did harrison work?")
    chain = setup_and_retrieval | prompt | model | output_parser

    print(chain.invoke("where did harrison work?"))


if __name__ == '__main__':
    # prompt_value = prompt.invoke({"topic": "ice cream"})
    # print(prompt_value)
    # print(prompt_value.to_messages())
    # print(prompt_value.to_string())
    # result = model.invoke(prompt_value)
    #
    # print(output_parser.invoke(result))
    # print(chain.invoke({"topic": "ice cream"}))
    rag_search_example()