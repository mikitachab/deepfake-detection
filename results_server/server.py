import datetime
from typing import List
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import mongoengine
import dotenv

app = FastAPI()
dotenv.load_dotenv()

MONGO_DB_ALIAS = os.getenv("MONGO_DB_ALIAS") or "results"
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME") or "results"
MONGO_DB_HOST = os.getenv("MONGO_DB_HOST") or "localhost"
MONGO_DB_PORT = os.getenv("MONGO_DB_PORT") or 27017
RESULTS_SECRET = os.getenv("RESULTS_SECRET") or ""
DEBUG = True if os.getenv("DEBUG") == "True" else False
DEFAULT_CONNECTION_NAME = mongoengine.connect(MONGO_DB_ALIAS)


def init_app():
    mongoengine.register_connection(
        alias=MONGO_DB_ALIAS, name=MONGO_DB_NAME, host=MONGO_DB_HOST, port=MONGO_DB_PORT
    )
    return app


class CVResult(BaseModel):
    cnn: str
    splits: List[float]
    preprocessing: str


class CVResultOut(BaseModel):
    cnn: str
    splits: List[float]
    preprocessing: str
    datetime: str


class CVResults(BaseModel):
    data: List[CVResultOut]


class DBCVResult(mongoengine.Document):
    datetime = mongoengine.DateTimeField(default=datetime.datetime.now)
    splits = mongoengine.ListField(mongoengine.FloatField())
    cnn = mongoengine.StringField()
    preprocessing = mongoengine.StringField()

    meta = {"collection": "results"}

    @staticmethod
    def from_model(result: CVResult):
        return DBCVResult(
            splits=result.splits, cnn=result.cnn, preprocessing=result.preprocessing
        )

    def to_model(self):
        return CVResultOut(
            cnn=self.cnn,
            splits=self.splits,
            preprocessing=self.preprocessing,
            datetime=self.datetime.strftime("%m.%d.%Y, %H:%M:%S"),
        )


@app.middleware("http")
async def check_secret(request: Request, call_next):
    if DEBUG:
        return await call_next(request)

    secret = request.headers.get("X-RESULTS-SECRET")
    if secret == RESULTS_SECRET:
        return await call_next(request)

    return JSONResponse(
        status_code=400,
        content={"message": "Who are you?"},
    )


@app.post("/result", response_model=CVResultOut, status_code=201)
async def add_result(result: CVResult):
    db_result = DBCVResult.from_model(result)
    db_result.save()
    return db_result.to_model()


@app.get("/result", response_model=CVResults, status_code=200)
async def get_results():
    return CVResults(data=[r.to_model() for r in DBCVResult.objects.all()])
