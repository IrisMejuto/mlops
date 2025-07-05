from fastapi import FastAPI
from class_values import UserData
import pandas as pd
import time
from transformers import pipeline

app = FastAPI()

@app.get("/saludo")
def saluda(name: str):
    return {"message": f"Hola {name}, tu asistente geoeconómico está en línea"}

@app.post("/user-info")
def register_user(user_data: UserData):
    time.sleep(2) 
    return {
        "message": f"Usuario {user_data.name} {user_data.surname} registrado.",
        "phone": user_data.phone_number,
        "address": user_data.address
    }

@app.get("/read-countries")
def read_dataframe():
    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv")
    return df.to_dict(orient="records")

@app.get("/get-gdp")
def country_gdp(country: str):
    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv")
    row = df[df["COUNTRY"].str.lower() == country.lower()]
    if row.empty:
        return {"error": f"País '{country}' no encontrado."}
    return {
        "country": row["COUNTRY"].values[0],
        "GDP (BILLIONS)": row["GDP (BILLIONS)"].values[0]
    }


@app.get("/paraphrase")
def paraphrase_text(text: str):
    pipe = pipeline("text2text-generation", model="ramsrigouthamg/t5_paraphraser")
    result = pipe(f"paraphrase: {text}", max_length=100)
    return {"paraphrased": result[0]["generated_text"]}



@app.get("/translate")
def translate_text(text: str):
    pipe = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")
    result = pipe(text, max_length=200)
    return {"translation": result[0]["translation_text"]}
