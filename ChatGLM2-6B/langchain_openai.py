import os
from langchain import OpenAI, ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from fastapi import FastAPI
import uvicorn
import json

app = FastAPI()

os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxx"

api_key = os.environ["OPENAI_API_KEY"]
llm = OpenAI()

# create the memory block
memory = ConversationBufferWindowMemory(k=5)
# memory.save_context({"input": "what's your name"}, {"output": "I am a bot from openai"})

# check the memory
last_token = memory.load_memory_variables({})
print(last_token)


def qa(query):
    conversation_with_summary = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    # response = llm.generate(
    #     prompts=[f"{query}"],
    #     model="text-davinci-003",
    #     api_type="open_ai"
    # )
    response = conversation_with_summary.predict(input=f"{query}")

    # save the answer to memory
    memory.save_context({"input": f"{query}"}, {"output": f"{response}"})
    return response


# query = input("Enter query:")
# print(query)
# ans = qa({"query": query})


# save the answer to memory
# memory.save_context({"input": f"{query}"}, {"output": f"{ans}"})


@app.get("/qa")
def qa_route(query: str):
    return {"answer": qa({"query": query})}


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
