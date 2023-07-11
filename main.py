import numpy as np
import pickle
from flask import Flask, url_for, render_template, request, redirect, session, jsonify
from flask_sqlalchemy import SQLAlchemy
import base64
import tensorflow as tf
from PIL import Image 
import cv2
import io
from keras.models import load_model
import h5py



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)


heart_model = pickle.load(open('model/heart_disease_model.pkl','rb'))
covid_model = load_model('model/covid.model')
liver_model = pickle.load(open('model/liver_model.pkl','rb'))
asd_model = pickle.load(open('model/asd_model.pkl','rb'))
diabetes_model = pickle.load(open('model/diabetes_model.pkl','rb'))


label_dict={0:'Covid19 Negative', 1:'Covid19 Positive'}
img_size = 100
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def init(self, username, password):
        self.username = username
        self.password = password


def predict(model,values,dic):
    values = np.asarray(values)
    return model.predict(values.reshape(1, -1))[0]


def preprocess(img):

	img=np.array(img)

	if(img.ndim==3):
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	else:
		gray=img

	gray=gray/255
	resized=cv2.resize(gray,(img_size,img_size))
	reshaped=resized.reshape(1,img_size,img_size)
	return reshaped

#url
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        u = request.form['username']
        p = request.form['password']
        data = User.query.filter_by(username=u, password=p).first()
        if data is not None:
            session['logged_in'] = True
            return redirect(url_for('main'))
        return render_template('login.html', message="Incorrect Details")



@app.route('/main')
def main():
    return render_template("main.html")

@app.route('/heart_disease')
def heart():
    return render_template("heart.html")

@app.route('/liver_disese')
def liver():
    return render_template("liver.html")

@app.route("/covid")
def covid():
	return(render_template("covid.html"))

@app.route("/diabetes")
def diabetes():
	return(render_template("diabetes.html"))

@app.route("/ASD")
def ASD():
	return(render_template("ASD.html"))

@app.route('/register/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            password=request.form['password']
            cpassword=request.form['cpassword']
            if password==cpassword:
                db.session.add(User(username=request.form['username'], password=request.form['password']))
                db.session.commit()
                return redirect(url_for('login'))
            
            else:
                return redirect(url_for('register'))

            
            
        except:
            return render_template('register.html', message="User Already Exists")
    else:
        return render_template('register.html')
    
@app.route("/predict_heart",methods= ['POST', 'GET'])
def predictPage_heart():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(heart_model,to_predict_list, to_predict_dict)
    except:
        message = "Please enter valid Data"
        return render_template("heart.html", message = message)

    return render_template('predict_heart.html', pred = pred)

@app.route("/predict_liver",methods= ['POST', 'GET'])
def predictPage_liver():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(liver_model,to_predict_list, to_predict_dict)
    except:
        message = "Please enter valid Data"
        return render_template("liver.html", message = message)

    return render_template('predict_liver.html', pred = pred)

@app.route("/predict_ASD",methods= ['POST', 'GET'])
def predictPage_ASD():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            print(to_predict_dict)
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            print(to_predict_list)
            pred = predict(asd_model,to_predict_list, to_predict_dict)
    except:
        message = "Please enter valid Data"
        return render_template("ASD.html", message = message)

    return render_template('predict_ASD.html', pred = pred)

@app.route("/predict_diabetes",methods= ['POST', 'GET'])
def predictPage_diabetes():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            print(to_predict_list)
            pred = predict(diabetes_model,to_predict_list, to_predict_dict)
            print(pred)
    except:
        message = "Please enter valid Data"
        return render_template("diabetes.html", message = message)

    return render_template('predict_diabetes.html', pred = pred)

@app.route("/predict_covid", methods=["POST"])
def predict_covid():
	print('HERE')
	message = request.get_json(force=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	dataBytesIO=io.BytesIO(decoded)
	dataBytesIO.seek(0)
	image = Image.open(dataBytesIO)

	test_image=preprocess(image)

	prediction = covid_model.predict(test_image)
	result=np.argmax(prediction,axis=1)[0]
	accuracy=float(np.max(prediction,axis=1)[0])

	label=label_dict[result]

	print(prediction,result,accuracy)

	response = {'prediction': {'result': label,'accuracy': accuracy}}

	return jsonify(response)

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session['logged_in'] = False
    return redirect(url_for('login'))

@app.route('/aboutus')
def aboutus():
    return render_template("aboutus.html")





if __name__ == '__main__':
    app.secret_key = "ThisIsNotASecret:p"
    with app.app_context():
        db.create_all()
        app.run(debug=True,port=5002)