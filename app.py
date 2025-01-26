from flask import Flask,render_template,request,jsonify
from flask_cors import CORS
app = Flask(__name__)
import pandas as pd
import numpy as np
import pickle
pipe=pickle.load(open("DiseasePredictionModel.pkl","rb"))

df=pd.read_csv("cleaned.csv")
CORS(app)

predictedtest=[]

leng=df.shape[0]
re=[]
for i in range(leng):
    li=[word.replace('_', ' ') for word in df.iloc[i,2].split()]
    for j in li:
        if j not in re:
            re.append(j)


@app.route("/getAllSymptoms")
def getAllSymptoms():
    return jsonify({"Symptoms": re})

@app.route("/")
def show():
    return render_template("index.html",symptoms=re)

@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data.get("symptoms", [])
    if symptoms:
        li = []
        for i in symptoms:
            li.append(i.replace(" ", "_"))
        y = " ".join(li)

        ypred = pipe.predict_proba([y])

        # Get the top 1 disease index based on probabilities
        top_1_index = np.argsort(ypred[0])[::-1][:1]

        # Get the top disease and its probability
        top_disease = pipe.classes_[top_1_index[0]]
        top_probability = ypred[0][top_1_index[0]]

        # Fetch disease information from the DataFrame
        disease_info = df[df['Disease'] == top_disease].iloc[0]
        description = disease_info['Description']
        precautions = disease_info['Precautions']

        # Create structured data for the response
        table_data = {
            "disease": top_disease,
            "description": description,
            "precautions": precautions
        }

        return jsonify({"success": True, "predictions": table_data})
    else:
        return jsonify({"success": False, "message": "No symptoms provided"})


if __name__ == "__main__":
    app.run(debug=True)
