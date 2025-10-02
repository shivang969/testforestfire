from flask import Flask,render_template,jsonify,request
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
application=Flask(__name__)
app=application
lassocv=pickle.load(open('env/first_end_to_end_project/models/ev.pkl','rb'))
sc=pickle.load(open('env/first_end_to_end_project/models/sc.pkl','rb'))
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
       day=float(request.form.get('day'))
       month=float(request.form.get('month'))
       RH=float(request.form.get('RH'))
       Ws=float(request.form.get('Ws'))
       Rain=float(request.form.get('Rain'))
       FFMC=float(request.form.get('FFMC'))
       DMC=float(request.form.get('DMC'))
       DC=float(request.form.get('DC'))
       ISI=float(request.form.get('ISI'))
       BUI=float(request.form.get('BUI'))
       Classes=float(request.form.get('Classes'))
       new_data=sc.transform([[day,month,RH,Ws,Rain,FFMC,DMC,DC,ISI,BUI,Classes]])
       prediction_text=lassocv.predict(new_data)
       return render_template('home.html',prediction_text=prediction_text[0])
    else:
        return render_template('home.html')
    
if __name__=="__main__":
    app.run(host="0.0.0.0")
    