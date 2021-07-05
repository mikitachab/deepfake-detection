import pytest

import mongoengine
from fastapi.testclient import TestClient

import server


@pytest.fixture(autouse=True)
def db_setup():
    mongoengine.disconnect()
    db_name = "mongoenginetest"
    conn = mongoengine.connect(
        db=db_name, host="mongomock://localhost", alias="default"
    )
    yield
    conn.drop_database(db_name)
    mongoengine.disconnect()


@pytest.fixture
def secret():
    s = "123"
    server.RESULTS_SECRET = s
    return s


@pytest.fixture
def client():
    return TestClient(server.app)


def test_can_add_result(client, secret):
    response = client.post(
        "/result",
        json={
            "cnn": "resnet18",
            "preprocessing": "preprocessing_pipeline",
            "splits": [0.98, 0.97, 0.98, 0.97, 0.85],
            "description": "first exp",
            "rnn_hidden_size": 3,
            "rnn_num_layers": 4
        },
        headers={"X-RESULTS-SECRET": secret},
    )
    assert response.status_code == 201

    data = response.json()
    assert data["cnn"] == "resnet18"
    assert data["preprocessing"] == "preprocessing_pipeline"
    assert data["splits"] == [0.98, 0.97, 0.98, 0.97, 0.85]
    assert data["description"] == "first exp"
    assert data["rnn_hidden_size"] == 3
    assert data["rnn_num_layers"] == 4

    assert "datetime" in data.keys()


def test_can_get_results(client, secret):
    response = client.post(
        "/result",
        json={
            "cnn": "resnet18",
            "preprocessing": "preprocessing_pipeline",
            "splits": [0.98, 0.97, 0.98, 0.97, 0.85],
            "description": "first exp",
            "rnn_hidden_size": 3,
            "rnn_num_layers": 4

        },
        headers={"X-RESULTS-SECRET": secret},
    )
    assert response.status_code == 201
    response = client.post(
        "/result",
        json={
            "cnn": "resnet18",
            "preprocessing": "no_preprocessing",
            "splits": [0.98, 0.97, 0.98, 0.97, 0.85],
            "description": "second exp",
            "rnn_hidden_size": 3,
            "rnn_num_layers": 4
        },
        headers={"X-RESULTS-SECRET": secret},
    )
    assert response.status_code == 201
    response = client.get("/result", headers={"X-RESULTS-SECRET": secret})

    assert response.status_code == 200

    data = response.json()["data"]
    assert len(data) == 2
