from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from transformers import T5ForConditionalGeneration, T5TokenizerFast

app = FastAPI()

templates = Jinja2Templates(directory="templates")


def load_model():
    MODEL_NAME = "danfarh2000/t5-base-end2end-chatbot-generative"
    checkpoint = "t5-base"
    chatbot_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer = T5TokenizerFast.from_pretrained(checkpoint)
    return tokenizer, chatbot_model


def run_model(input_string, **generator_args):
  generator_args = {
  "max_length": 256,
  "num_beams": 4,
  "length_penalty": 1.5,
  "no_repeat_ngram_size": 3,
  "early_stopping": True,
  }
  tokenizer, chatbot_model = load_model()
  input_string = input_string
  input_ids = tokenizer.encode(input_string, return_tensors="pt")
  res = chatbot_model.generate(input_ids, **generator_args)
  output = tokenizer.batch_decode(res, skip_special_tokens=True)
  return output


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/{question}")
async def response(question: str):
    response = run_model(question)
    x = slice(9, -7)
    print(response[0][x])
    return {'response': response[0][x]}