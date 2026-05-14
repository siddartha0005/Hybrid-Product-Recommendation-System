from flask import Flask, request, render_template, redirect, url_for, flash, session
import pandas as pd
import os
import random
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
# Initialize Flask app
app = Flask(__name__)

# Database configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'mysecret')  # Secret key for session handling
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:6174%40Venkat@localhost/ecom"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

# Define User model
class SearchHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete="CASCADE"), nullable=False)
    product_name = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

    user = db.relationship('User', backref=db.backref('search_history', lazy=True))



# Load data (ensure CSV files exist in 'models' folder)
try:
    trending_products = pd.read_csv("models/trending_products.csv")
    train_data = pd.read_csv("models/clean_data.csv")
except Exception as e:
    print(f"Error loading CSV files: {e}")
    trending_products = pd.DataFrame()
    train_data = pd.DataFrame()


# Function to truncate text
def truncate(text, length):
    return text[:length] + "..." if len(text) > length else text


# Function to clean image URLs
def clean_image_url(url):
    if pd.isna(url):
        return None
    urls = url.split(' | ')
    for img in urls:
        if img.startswith("http"):
            return img  # Return the first valid image URL
    return None


# Content-based recommendation function
def content_based_recommendations(train_data, item_name, top_n=10):
    matched_product, score = process.extractOne(item_name, train_data['Name'].values)
    if matched_product is None or score < 70:
        return pd.DataFrame()

    item_index = train_data[train_data['Name'] == matched_product].index[0]
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    similar_items = list(enumerate(cosine_similarities_content[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    recommended_item_indices = [x[0] for x in similar_items[1:top_n + 1]]

    recommendations = train_data.iloc[recommended_item_indices].copy()
    recommendations['ImageURL'] = recommendations['ImageURL'].apply(clean_image_url)
    recommendations = recommendations.dropna(subset=['ImageURL'])  # Remove rows with no valid image URL

    return recommendations[['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]


# Routes
@app.route('/evaluate')
def evaluate():
    # You need some ground truth and predicted data
    # For demonstration, let's simulate:
    # 1 = relevant/recommended correctly, 0 = not relevant

    # Simulated dummy data
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]  # Actual values
    y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1]  # Predicted values from recommendations

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()  # Convert to list for Jinja compatibility

    return render_template('evaluation.html',
                           precision=round(precision, 2),
                           recall=round(recall, 2),
                           f1_score=round(f1, 2),
                           accuracy=round(accuracy, 2),
                           confusion_matrix=cm)
@app.route("/")
def index():
    recommended_products = pd.DataFrame()
    username = None  # Default value

    if 'user_id' in session:
        user_id = session['user_id']

        # Fetch the logged-in user's details
        user = User.query.get(user_id)
        if user:
            username = user.username  # Get username

        # Fetch user's search history & recommendations
        recent_searches = SearchHistory.query.filter_by(user_id=user_id).order_by(SearchHistory.timestamp.desc()).limit(
            5).all()
        search_terms = [search.product_name for search in recent_searches]
        user_recommendations = []

        for term in search_terms:
            recs = content_based_recommendations(train_data, term, top_n=3)
            user_recommendations.append(recs)

        if user_recommendations:
            recommended_products = pd.concat(user_recommendations).drop_duplicates().head(10)
        else:
            recommended_products = trending_products.head(8)

    else:
        recommended_products = trending_products.head(8)

    recommended_products['ImageURL'] = recommended_products['ImageURL'].apply(clean_image_url)
    recommended_products = recommended_products.dropna(subset=['ImageURL'])
    recommended_products = recommended_products.to_dict(orient='records')

    return render_template('index.html', recommended_products=recommended_products, truncate=truncate,
                           username=username)


@app.route("/main", methods=['GET'])
def main():
    content_based_rec = pd.DataFrame()
    return render_template('main.html', content_based_rec=content_based_rec)


@app.route("/signup", methods=['POST'])
def signup():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']

    # ✅ Ensure User model is defined before querying
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        flash("Email already registered. Please log in.", "danger")
        return redirect(url_for("index"))

    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
    new_user = User(username=username, email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    flash("Sign-up successful! Please log in.", "success")
    return redirect(url_for("index"))


@app.route('/signin', methods=['POST'])
def signin():
    if 'email' not in request.form or 'password' not in request.form:
        flash("Missing email or password. Please try again.", "danger")
        return redirect(url_for("index"))

    email = request.form['email']
    password = request.form['password']

    user = User.query.filter_by(email=email).first()
    if user and check_password_hash(user.password, password):
        session['user_id'] = user.id
        flash("Login successful!", "success")
        return redirect(url_for("index"))
    else:
        flash("Invalid email or password. Try again.", "danger")
        return redirect(url_for("index"))


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("Logged out successfully!", "success")
    return redirect(url_for("index"))


@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        prod = request.form.get('prod')
        nbr = int(request.form.get('nbr'))

        # Save search if user is logged in
        if 'user_id' in session:
            user_id = session['user_id']
            search_entry = SearchHistory(user_id=user_id, product_name=prod)
            db.session.add(search_entry)
            db.session.commit()

        # Get product recommendations
        content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)

        return render_template('main.html', content_based_rec=content_based_rec, searched_product=prod, searched_count=nbr, truncate=truncate)

    return render_template('main.html', content_based_rec=pd.DataFrame(), message="Please enter a product name.")




if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
