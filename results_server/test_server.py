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


@pytest.fixture
def fake_splits():
    return [
        {
            'train': {'accuracy': 0.668, 'precision': 0.697, 'recall': 0.596}, 
            'test': {'accuracy': 0.689, 'precision': 0.634, 'recall': 0.892}
        }
    ]

def test_can_add_result(client, secret, fake_splits):
    response = client.post(
        "/result",
        json={
            "cnn": "resnet18",
            "preprocessing": "preprocessing_pipeline",
            "splits": fake_splits,
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
    assert data["splits"] == fake_splits
    assert data["description"] == "first exp"
    assert data["rnn_hidden_size"] == 3
    assert data["rnn_num_layers"] == 4

    assert "datetime" in data.keys()


def test_can_get_results(client, secret, fake_splits):
    response = client.post(
        "/result",
        json={
            "cnn": "resnet18",
            "preprocessing": "preprocessing_pipeline",
            "splits":fake_splits,
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
            "splits": fake_splits,
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
