from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import requests
from io import BytesIO

from app.model import generate_embedding

app = FastAPI()


class ImageRequest(BaseModel):
    imageUrl: str


@app.get("/health")
def health():
    return {"status": "alive"}
    

@app.post("/embed")
def embed_face(req: ImageRequest):

    response = requests.get(req.imageUrl)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    embedding = generate_embedding(image)

    if embedding is None:
        return {"error": "No face detected"}

    return {"embedding": embedding}
