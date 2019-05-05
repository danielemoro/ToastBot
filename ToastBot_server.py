import keras
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from ToastBot_predict import *

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

current_compliment = "None"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/compliment', methods=['GET', 'POST'])
def generate_compliment():
    global current_compliment
    return '''
        <!doctype html>
        <title>ToastBot</title>
        <h1>''' + current_compliment + '''</h1>
        <a href="./"> Go back </a>
        '''

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # GET TEXT
        curr_text = request.form['text']

        # GET IMAGE
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            global current_compliment
            current_compliment = get_compliment(curr_text, file_path)

            return redirect(url_for('generate_compliment',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>ToastBot</title>
    <h1> Welcome to ToastBot! </h1>
    <form method=post enctype=multipart/form-data>
      <b>How are you feeling?</b> <input name="text">
      <p>
      <b> Upload your picture: </b> <input type=file name=file>
      <p>
      <input type=submit value="Generate Compliment">
    </form>
    '''

if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'

    app.run()
