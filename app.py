from flask import Flask, jsonify
import pandas as pd
import tensorflow as tf
from process import load_songs, preprocess_text, build_song_model, define_task, create_retrieval_model, \
    setup_retrieval_model, get_song_recommendations

app = Flask(__name__)

# Load the data and initialize the model
songs = load_songs()
songs = preprocess_text(songs)
song_model = build_song_model()
task = define_task(songs, song_model)
model = create_retrieval_model(song_model, task)
index = setup_retrieval_model(model, songs)

# Load the data into the 'data' DataFrame
data = pd.read_csv('Untrimmeddata.csv')
try:
    saved_model = tf.keras.models.load_model('content_based_model.h5')
except OSError as e:
    print(f"Failed to load the model: {e}")

# Define a route for the root URL
@app.route('/')
def hello():
    return "Hello, Flask!"

@app.route('/recommend/<int:song_id>', methods=['GET'])
def recommend_song(song_id):
    try:
        track = data[data['id'] == song_id]['Track'].values[0]  # Retrieve the associated track
        recommendations, urls = get_song_recommendations(index, song_id, data)

        response = {
            'Input Song ID': song_id,
            'Input Song Name': track,
            'Output Song ID': recommendations.tolist(),
            'Output Song URLs': urls
        }
        return jsonify(response), 200  # Return a successful response
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return jsonify({'error': error_message}), 500  # Return an error response


if __name__ == '__main__':
    app.run()