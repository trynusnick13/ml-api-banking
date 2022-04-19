import warnings

from fastapi import FastAPI

from transformations import transform_user_details_to_scalars
from user_model import User
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

warnings.filterwarnings("ignore")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
async def ping():
    return {"message": "pong"}


@app.post("/predict/")
async def predict(user: User):
    print(user.dict())
    result = transform_user_details_to_scalars(user.dict())
    response = "Yes" if result else "No"

    return {"message": response}
