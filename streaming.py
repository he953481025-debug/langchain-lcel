# Showing the example using anthropic, but you can use
# your favorite chat model!


if __name__ == '__main__':

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