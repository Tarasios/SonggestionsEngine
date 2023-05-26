# SonggestionsEngine
This repository contains the code for Songgestions, a web application that uses Machine Learning to recommend songs.

## Technologies Used
* Backend: Node.js, Flask
* Machine Learning Framework: TensorFlow
* Data Processing: pandas

## File Contents
* app.py: The main Python file that runs the web application.
* content_based_model.h5: Pre-trained TensorFlow model for song recommendations.
* process.py: Python script for data processing.
* Procfile: Configuration file for deploying the application.
* requirements.txt: List of Python dependencies required for the project.
* Untrimmeddata.csv: Kaggle dataset used for training the model.

## Installation and Setup
To set up the development environment for Songgestions, follow these steps:

1. Clone this GitHub repository: git clone https://github.com/Tarasios/Songgestions.git
2. Navigate to the project folder: cd Songgestions
3. Set up a Python environment (e.g., using virtualenv or conda).
4. Install the required dependencies: pip install -r requirements.txt
5. Run the application: python app.py
6. Open your web browser and go to the Songgestions website http://pqrhigvnxy.eu09.qoddiapp.com/.

## How to Use
To use Songgestions:

1. Visit the Songgestions website http://pqrhigvnxy.eu09.qoddiapp.com/.
2. Search for a song using the provided search functionality.
3. Click the "Songgest" button to receive recommendations for similar songs.

## Credits, References, and Licenses
* The Kaggle dataset used in this project is sourced from https://www.kaggle.com/datasets/salvatorerastelli/spotify-and-youtube
* This API developed by Tarasios as part of the Songgestions 2800 project in collaboration with Quincy, Connie, and Vincent.

## AI Usage
In Songgestions, we utilized Machine Learning to power our song recommendation system. Here's how AI was used:

* We built our own TensorFlow model for song recommendations using the provided Kaggle dataset.
* ChatGPT, an AI language model, assisted us in creating a roadmap for our project and guiding us through certain decision points.
* As the web server hosting limitations prevent us from running the Python API server-side, the TensorFlow model needs to be downloaded and run client-side.

## Contact Information
For any questions or collaboration opportunities, please contact Tarasios on Discord at Tarasios#9030.
