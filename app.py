from flask import Flask,render_template,request,redirect
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os 
import numpy as np
model=load_model('face_model.h5',compile=False)

app=Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/",methods=['GET','POST'])
def main():
    Age=None
    Gender=None
    img_path=None
    if request.method=='POST':
        file=request.files["fileInput"]
        if file:
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(img_path)
            img=image.load_img(img_path,target_size=(200,200))
            img=image.img_to_array(img)
            img=img/255
            img=np.expand_dims(img,axis=0)
            pred=model.predict(img)
            Age=pred[0][0]
            if pred[1][0]<=0.5:
                Gender="Male"
            else:
                Gender="Female"
            




    return render_template('base.html',
                           Gender=Gender,
                           Age=Age,
                           img_path=img_path)


if __name__=="__main__":
    app.run(debug=True)