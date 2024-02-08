from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

app = Flask(__name__)

insta_data = pd.read_csv('insta_train.csv')
insta_X = insta_data.drop('fake', axis=1)
insta_y = insta_data['fake']
insta_X_train, _, insta_y_train, _ = train_test_split(insta_X, insta_y, test_size=0.2, random_state=42)

insta_model = Pipeline(steps=[
    ('preprocessor',ColumnTransformer(transformers=[
                ('num', Pipeline(steps=[('scaler', StandardScaler())]), 
                    ['profile pic', 'nums/length username', 'fullname words', 'nums/length fullname', 'name==username', 'description length', 'external URL', '#posts', '#followers', '#follows']),
                ('cat', Pipeline(steps=[('onehot', OneHotEncoder())]), ['private'])
            ])
        ),
    ('classifier', RandomForestClassifier())
])
insta_model.fit(insta_X_train, insta_y_train)

# Load Twitter dataset and train the model
twitter_data = pd.read_csv('twitter_data.csv')
twitter_X = twitter_data.drop(['UserID', 'Fake Or Not Category'], axis=1)
twitter_y = twitter_data['Fake Or Not Category']
_, twitter_X_train, _, twitter_y_train = train_test_split(twitter_X, twitter_y, test_size=0.8, random_state=42)

twitter_model = RandomForestClassifier()
twitter_model.fit(twitter_X_train, twitter_y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/insta')
def insta():
    return render_template('index_insta.html')

@app.route('/twitter')
def twitter():
    return render_template('index_twitter.html')

@app.route('/insta_predict', methods=['POST'])
def insta_predict():
    if request.method == 'POST':
        # Get user inputs from the form
        new_data = {
            'profile pic': int(request.form['profile_pic']),
            'nums/length username': float(request.form['nums_length_username']),
            'fullname words': int(request.form['fullname_words']),
            'nums/length fullname': float(request.form['nums_length_fullname']),
            'name==username': int(request.form['name_eq_username']),
            'description length': int(request.form['description_length']),
            'external URL': int(request.form['external_URL']),
            '#posts': int(request.form['num_posts']),
            '#followers': int(request.form['num_followers']),
            '#follows': int(request.form['num_follows']),
            'private': int(request.form['account_private'])
        }

        # Convert the input data into a DataFrame
        new_df = pd.DataFrame([new_data])

        # Make predictions using the trained model
        prediction = insta_model.predict(new_df)

        # Map numerical prediction to label
        result = "Fake" if prediction[0] == 1 else "Real"

        # Render the appropriate result page based on the prediction
        if result == "Real":
            return render_template('insta_result_real.html')
        else:
            return render_template('insta_result_fake.html')

@app.route('/twitter_predict', methods=['POST'])
def twitter_predict():
    if request.method == 'POST':
        new_data = {
            'No Of Abuse Report': int(request.form['abuse_reports']),
            'No Of Rejected Friend Requests': int(request.form['rejected_friend_requests']),
            'No Of Freind Requests Thar Are Not Accepted': int(request.form['unaccepted_friend_requests']),
            'No Of Friends': int(request.form['friends']),
            'No Of Followers': int(request.form['followers']),
            'No Of Likes To Unknown Account': int(request.form['likes_to_unknown_accounts']),
            'No Of Comments Per Day': int(request.form['comments_per_day'])
        }

        new_df = pd.DataFrame([new_data])
        prediction = twitter_model.predict(new_df)
        result = "Fake" if prediction[0] == 1 else "Real"

       # Render the appropriate result page based on the prediction
        if result == "Real":
            return render_template('twitter_result_real.html')
        else:
            return render_template('twitter_result_fake.html')

if __name__ == '__main__':
    app.run(debug=True)
