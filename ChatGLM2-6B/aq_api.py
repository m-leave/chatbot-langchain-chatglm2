from main import GLM
from fastapi import FastAPI
import uvicorn

app = FastAPI()

# # Load the local ChatGLM2 model and answer questions
model = GLM()
model.load_model()


# response = model.generate(prompts=["告诉我什么是ai"])
# print(response)


def answer_questions(question):
    return model.generate(prompts=[f"{question}"])


@app.get("/qa")
def qa_route(question: str):
    return {"answer": answer_questions(question)}


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
