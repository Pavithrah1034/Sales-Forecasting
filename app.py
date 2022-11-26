from flask import Flask, request
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from model import model

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = "C:/Users/Pavithrah/Desktop/sales prediction/dataset/"

@app.route('/upload' , methods = ['GET','POST'])
def onUpload():
    if request.method == 'POST':
        model()
        f = request.files['files']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #title = request.form['title']
        #content = request.form['content']
        #stored_filename = './dataset/'
        #stored_filename = filename
        #print(stored_filename)
        #predict = model(stored_filename, title, content)
        #os.remove(os.path.join(app.config["UPLOAD_FOLDER"],filename))
        #print(predict)
    return 'File uploaded'

    
if __name__ == '_main_':
    app.run(debug = True)
