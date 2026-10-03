import _path  # noqa
from underwriting import credit_model

print(credit_model.train(log_mlflow=True))
