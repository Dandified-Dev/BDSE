import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sqlalchemy import create_engine

# Database connection
engine = create_engine('mysql+pymysql://root:DandiVojuke74!@localhost/onboarding')

# Load and preprocess training data from the database
query = "SELECT * FROM train WHERE root_genre = 'Jazz'"
df = pd.read_sql(query, engine)
df.drop_duplicates(subset=df.columns, keep='first', inplace=True)
# Combine text columns into a single 'text' column
df['text'] = df[['reviewText', 'summary', 'overall']].astype(str).agg(' '.join, axis=1)

# Convert to lowercase
df['text'] = df['text'].apply(lambda x: x.lower())

# Remove non-alphabetic characters and extra whitespaces
df['text'] = df['text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

# Tokenization (split the text into words)
df['text'] = df['text'].apply(lambda x: x.split())

# Remove stopwords
stop_words = set(stopwords.words('english'))
df['text'] = df['text'].apply(lambda x: [word for word in x if word not in stop_words])

# Stemming
ps = PorterStemmer()
df['text'] = df['text'].apply(lambda x: [ps.stem(word) for word in x])

# Convert tokens back to a single string
df['text'] = df['text'].apply(lambda x: ' '.join(x))

# Define a function to get sentiment polarity using TextBlob
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Apply sentiment analysis to each row in the DataFrame
df['polarity'] = df['text'].apply(get_sentiment)

# Add a new feature for the length of the reviews
df['review_length'] = df['text'].apply(lambda x: len(x.split()))

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['polarity']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection and training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# Add a new column indicating positive or negative sentiment
df['sentiment'] = df['polarity'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')

# Create Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div(children=[
    html.H1(children='Sentiment Analysis and Review Length Dashboard'),

    # Display DataFrame
    html.Div([
        html.H2('DataFrame with Sentiment Polarity, Sentiment, and Review Length'),
        dcc.Textarea(
            id='dataframe-textarea',
            value=df[['text', 'polarity', 'sentiment', 'review_length']].to_string(),
            style={'width': '100%', 'height': 300},
            readOnly=True
        )
    ]),

    # Display Model Evaluation Metrics
    html.Div([
        html.H2('Model Evaluation Metrics'),
        html.P(f'Mean Squared Error: {mse:.2f}'),
        html.P(f'R-squared Score: {r2:.2f}')
    ]),

    # Display Histogram
    html.Div([
        html.H2('Distribution of Review Lengths'),
        dcc.Graph(
            id='histogram',
            figure={
                'data': [
                    {'x': df['review_length'], 'type': 'histogram', 'name': 'Review Lengths'}
                ],
                'layout': {
                    'title': 'Distribution of Review Lengths',
                    'xaxis': {'title': 'Review Length'},
                    'yaxis': {'title': 'Frequency'}
                }
            }
        )
    ]),

    # Display Scatter Plot
    html.Div([
        html.H2('Scatter plot of Sentiment Polarity vs. Review Length'),
        dcc.Graph(
            id='scatter-plot',
            figure={
                'data': [
                    {'x': df[df['sentiment']=='Positive']['polarity'], 'y': df[df['sentiment']=='Positive']['review_length'], 'mode': 'markers', 'name': 'Positive Sentiment'},
                    {'x': df[df['sentiment']=='Negative']['polarity'], 'y': df[df['sentiment']=='Negative']['review_length'], 'mode': 'markers', 'name': 'Negative Sentiment'}
                ],
                'layout': {
                    'title': 'Scatter plot of Sentiment Polarity vs. Review Length',
                    'xaxis': {'title': 'Sentiment Polarity'},
                    'yaxis': {'title': 'Review Length'}
                }
            }
        )
    ])
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)