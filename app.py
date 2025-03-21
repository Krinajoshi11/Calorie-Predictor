from flask import Flask, render_template, request, jsonify
import pickle
import os  # Import os module for environment variables

app = Flask(__name__)

# Load trained model & vectorizer
model = pickle.load(open("calorie_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        food_item = request.form['food_item'].strip()

        # Check if the input is empty
        if not food_item:
            return render_template('index.html', error="Please enter a food item!")

        # Convert food name into numerical features
        X = vectorizer.transform([food_item])

        # Check if input exists in the trained vocabulary
        if not X.nnz:  # If no valid features are extracted
            return render_template('index.html', error="Invalid item, cannot predict calories")

        # Predict calories
        predicted_calories = model.predict(X)[0]
        return render_template('index.html', prediction=round(predicted_calories, 2))

    return render_template('index.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Use Render's assigned port or default to 8080
    app.run(host="0.0.0.0", port=port)  # Change host to "0.0.0.0" to make it accessible online
