from pydantic import BaseModel

class PredictResults(BaseModel):
    churn_result: bool
    churn_probability: float