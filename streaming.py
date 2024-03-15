# Showing the example using anthropic, but you can use
# your favorite chat model!
import os

from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

from dotenv import load_dotenv

load_dotenv()


async def main():
    from langchain_community.chat_models import ChatOpenAI
    model = ChatOpenAI(openai_api_key="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")

    chunks = []
    async for chunk in model.astream("hello. tell me something about yourself"):
        chunks.append(chunk)
        print(chunk.content, end="|", flush=True)

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
    parser = StrOutputParser()
    chain = prompt | model | parser

    async for chunk in chain.astream({"topic": "parrot"}):
        print(chunk, end="|", flush=True)

    from langchain_core.output_parsers import JsonOutputParser

    chain = (
            model | JsonOutputParser()
    )  # Due to a bug in older versions of Langchain, JsonOutputParser did not stream results from some models
    async for text in chain.astream(
            'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`'
    ):
        print(text, flush=True)

    from langchain_core.output_parsers import JsonOutputParser

    async def _extract_country_names_streaming(input_stream):
        """A function that operates on input streams."""
        country_names_so_far = set()

        async for input in input_stream:
            if not isinstance(input, dict):
                continue

            if "countries" not in input:
                continue

            countries = input["countries"]

            if not isinstance(countries, list):
                continue

            for country in countries:
                name = country.get("name")
                if not name:
                    continue
                if name not in country_names_so_far:
                    yield name
                    country_names_so_far.add(name)

    chain = model | JsonOutputParser() | _extract_country_names_streaming

    async for text in chain.astream(
            'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`'
    ):
        print(text, end="|", flush=True)


async def non_streaming_component():
    # Function that does not support streaming.
    # It operates on the finalizes inputs rather than
    # operating on the input stream.
    def _extract_country_names(inputs):
        """A function that does not operates on input streams and breaks streaming."""
        if not isinstance(inputs, dict):
            return ""

        if "countries" not in inputs:
            return ""

        countries = inputs["countries"]

        if not isinstance(countries, list):
            return ""

        country_names = [
            country.get("name") for country in countries if isinstance(country, dict)
        ]
        return country_names

    model = ChatOpenAI()
    chain = (
            model | JsonOutputParser() | _extract_country_names
    )  # This parser only works with OpenAI right now

    async for chunk in chain.astream(
            'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`',
    ):
        print(chunk, flush=True)


# An LCEL chain constructed using non-streaming components, will still be able to stream in a lot of cases, with streaming
# of partial output starting after the last non-streaming step in the chain.
def retrieval_chain_stream():
    vectorstore = FAISS.from_texts(
        ["harrison worked at kensho", "harrison likes spicy food"],
        embedding=OpenAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever()
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    from langchain_community.chat_models import ChatOpenAI
    model = ChatOpenAI()
    retrieval_chain = (
            {
                "context": retriever.with_config(run_name="Docs"),
                "question": RunnablePassthrough(),
            }
            | prompt
            | model
            | StrOutputParser()
    )
    for chunk in retrieval_chain.stream(
            "Where did harrison work? " "Write 3 made up sentences about this place."
    ):
        print(chunk, end="|", flush=True)


async def astream_event():
    num_events = 0
    model = ChatOpenAI()
    chain = (
            model | JsonOutputParser()
    )  # Due to a bug in older versions of Langchain, JsonOutputParser did not stream results from some models

    async for event in chain.astream_events(
            'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`',
            version="v1",
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            print(
                f"Chat model chunk: {repr(event['data']['chunk'].content)}",
                flush=True,
            )
        if kind == "on_parser_stream":
            print(f"Parser chunk: {event['data']['chunk']}", flush=True)
        num_events += 1
        if num_events > 30:
            # Truncate the output
            print("...")
            break

    chain = model.with_config({"run_name": "model"}) | JsonOutputParser().with_config(
        {"run_name": "my_parser"}
    )
    # filter by name
    max_events = 0
    async for event in chain.astream_events(
            'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`',
            version="v1",
            include_names=["model"],
    ):
        print(event)
        max_events += 1
        if max_events > 10:
            # Truncate output
            print("...")
            break

    chain = model.with_config({"run_name": "model"}) | JsonOutputParser().with_config(
        {"run_name": "my_parser"}
    )

    # filter by type
    max_events = 0
    async for event in chain.astream_events(
            'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`',
            version="v1",
            include_types=["chat_model"],
    ):
        print(event)
        max_events += 1
        if max_events > 10:
            # Truncate output
            print("...")
            break
    # filter by tags
    chain = (model | JsonOutputParser()).with_config({"tags": ["my_chain"]})

    max_events = 0
    async for event in chain.astream_events(
            'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`',
            version="v1",
            include_tags=["my_chain"],
    ):
        print(event)
        max_events += 1
        if max_events > 10:
            # Truncate output
            print("...")
            break


async def propagating_callback():
    from langchain_core.runnables import RunnableLambda
    from langchain_core.tools import tool

    def reverse_word(word: str):
        return word[::-1]

    reverse_word = RunnableLambda(reverse_word)

    @tool
    def bad_tool(word: str):
        """Custom tool that doesn't propagate callbacks."""
        return reverse_word.invoke(word)

    @tool
    def correct_tool(word: str, callbacks):
        """A tool that correctly propagates callbacks."""
        return reverse_word.invoke(word, {"callbacks": callbacks})

    async for event in correct_tool.astream_events("hello", version="v1"):
        print(event)

    from langchain_core.runnables import RunnableLambda
    from langchain_core.runnables import chain
    @chain
    async def reverse_and_double(word: str):
        return await reverse_word.ainvoke(word) * 2

    # reverse_and_double = RunnableLambda(reverse_and_double)

    await reverse_and_double.ainvoke("1234")

    async for event in reverse_and_double.astream_events("1234", version="v1"):
        print(event)


if __name__ == '__main__':
    import asyncio

    asyncio.run(propagating_callback(), debug=True)
    # non_streaming_component()
    # retrieval_chain_stream()
