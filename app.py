import os
import warnings
import pandas as pd
from PIL import Image
from flask import Flask,  flash, request, redirect, url_for, render_template, session
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from werkzeug.utils import secure_filename

warnings.filterwarnings("ignore")

# Linear Regression for pH_value
df = pd.read_csv('ph-data.csv')
X = df[['R', 'G', 'B']]
y = df['pH']
pH_model = LinearRegression()
pH_model.fit(X, y)

# SVM algorithm to predict the crop
df_crop = pd.read_csv('Book1.csv')
X_crop = df_crop[['pH_values']]
y_crop = df_crop['Crop']
# Cluster the crop data using k-means
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_crop)

# Gradient Boosting to get NPK values
df = pd.read_csv('fertilizer_data.csv')
X_train, X_test, y_train, y_test = train_test_split(df[['pH','Crop']], df[['N', 'P', 'K']], test_size=0.2, random_state=42)
model_n = GradientBoostingRegressor()
model_n.fit(X_train, y_train['N'])
model_p = GradientBoostingRegressor()
model_p.fit(X_train, y_train['P'])
model_k = GradientBoostingRegressor()
model_k.fit(X_train, y_train['K'])
score_n = model_n.score(X_test, y_test['N'])
score_p = model_p.score(X_test, y_test['P'])
score_k = model_k.score(X_test, y_test['K'])

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
IMG_FOLDER = os.path.join('static', 'crops')

app.config['UPLOAD_FOLDER1'] = IMG_FOLDER
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def check():
    return render_template('index.html')

@app.route('/index', methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET','POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # Resize the image
            width, height = 700, 350
            img = img.resize((width, height))
            resized_filename = f"{filename}"
            img.save(os.path.join(app.config['UPLOAD_FOLDER'], resized_filename))
            flash('Image successfully uploaded and displayed below')
            return render_template('Predict.html', filename=resized_filename)
        else:
            flash('Allowed image types are - png, jpg, jpeg, gif')
            return redirect(request.url)
    else:
        return render_template('home.html')

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static',filename='uploads/'+filename),code=302)

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method=='POST':
        filename = request.form['filename']
        img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # get coordinates and filename from GET request
        x = int(request.form['x'])
        y = int(request.form['y'])
    # get pixel color at mouse position
        r, g, b = img.getpixel((x, y))
    # Predict the pH value using the trained linear regression model
        pH_value = pH_model.predict([[r, g, b]])
                # Assign the pH value to the nearest cluster
        cluster_label = kmeans.predict([[pH_value[0]]])[0]
        # Get the crops associated with the cluster
        crops = y_crop[kmeans.labels_ == cluster_label].unique()
        # Store the pH_value and crop_label in session
        session['pH_value'] = pH_value[0]
        crops = crops.tolist()
        session['crops'] = crops
       # Redirect to the result page
        return redirect(url_for('result'))

@app.route('/result')
def result():
    # Retrieve the pH_value and crop_label from session
    pH_value = session.get('pH_value')
    crops = session.get('crops',[])
    # Render the result template with the pH_value and crop_label
    return render_template('Result.html', pH_value=pH_value, crops=crops)

@app.route('/ferti')
def ferti():
    return render_template('Ferti.html')

@app.route('/fertilizer',methods=['POST'])
def fertilizer():
    pH_value=float(request.form['pH_value'])
    crop_label=int(request.form['Crop'])
    area = float(request.form['area'])
    new_data = pd.DataFrame({ 'pH': [pH_value],'Crop': [crop_label]})
    prediction_n = int(model_n.predict(new_data)[0])
    prediction_p = int(model_p.predict(new_data)[0])
    prediction_k = int(model_k.predict(new_data)[0])
    MOP = prediction_k*1.67*area
    DAP = prediction_p*2.17*area
    UN = DAP*0.18
    pred_n = prediction_n-UN
    U = pred_n*2.17*area
    Cost_U = U*5.94
    Cost_MOP = MOP*6.06
    Cost_DAP = DAP*10.25
    Total_cost = Cost_U+Cost_DAP+Cost_MOP
    return render_template('Fertilizer.html',N=prediction_n,P=prediction_p,K=prediction_k,U='%.3f'%U,MOP='%.3f'%MOP,DAP='%.3f'%DAP,CU='%.3f'%Cost_U,CDAP='%.3f'%Cost_DAP,CMOP='%.3f'%Cost_MOP,TC='%.3f'%Total_cost)

if __name__ == "__main__":
    app.run(debug=True)




