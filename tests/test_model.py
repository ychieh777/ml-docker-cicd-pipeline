from src.predict import predict

def test_prediction():
    result = predict(80, 0.03)
    assert result in [0, 1]