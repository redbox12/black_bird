from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel


class Item(BaseModel):
    question: str


app = FastAPI()

@app.get("/")
def read_root():
    return {"Здорово, бандиты!"}



@app.post("/question/")
async def create_item(item: Item):
    return item.question