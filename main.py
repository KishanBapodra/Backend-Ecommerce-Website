from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel

class churn_request(BaseModel):
    service : int
    ui : int
    price : int

# Declaring our FastAPI instance
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*']
)

with open('dtree.pkl','rb') as f:
    model = joblib.load(f)

# Endpoint to return prediction based on input values
@app.post('/predict/churn')
def predictChurn(data: churn_request):
    df = [[
        data.service,
        data.ui,
        data.price
    ]]
    pred = model.predict(df)
    return {"prediction": int(pred)}

# Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'Welcome to Ecomm-backend!'}