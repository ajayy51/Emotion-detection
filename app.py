from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the trained model
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)


# Define the home route
@app.route('/')
def home():
    return render_template('index.html')



# route to predict sentiment for a single feedback
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        feedback = request.form['feedback']
        prediction = model.predict([feedback])[0]  # Predict the emotion
        return render_template('index.html', prediction=f'The emotion is: {prediction}')


# route to handle CSV upload and generate chart
@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    chart_type = request.form['chart']

    # Read the CSV file into a DataFrame
    data = pd.read_csv(file)

    # Check if 'Feedback' column exists
    if 'Feedback' not in data.columns:
        return render_template('index.html', prediction="CSV must contain 'Feedback' column.")

    # Predict emotions for each feedback in the CSV
    data['Predicted_Emotion'] = model.predict(data['Feedback'])

    # Count the occurrences of each emotion
    emotion_counts = data['Predicted_Emotion'].value_counts()

    # Plot the emotion distribution
    img = io.BytesIO()
    plt.figure(figsize=(8, 6))
    if chart_type == 'bar':
        emotion_counts.plot(kind='bar', color='skyblue')
        plt.title('Emotion Distribution - Bar Chart')
        plt.xlabel('Emotion')
        plt.ylabel('Count')
    else:
        emotion_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140,
                            colors=['skyblue', 'lightgreen', 'orange', 'pink', 'lightcoral'])
        plt.title('Emotion Distribution - Pie Chart')
        plt.ylabel('')

    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()

    # Render the result with the generated chart
    return render_template('index.html', chart_url=f"data:image/png;base64,{chart_url}")


if __name__ == '__main__':
    app.run(debug=True)
