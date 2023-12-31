# Weather Prediction with Scikit Learn, Streamlit and Deployed with Flask

![](./assets/prev-1.gif)

## Weather Prediction with Scikit Learn, Numpy, Pandas, Streamlit and Deployed with Flask 

The Model was trained with Tabular Weather Data and with the `KNeighborsClassifier` Scikit-Learn Architecture. The Model predicts if a Weather will be either `Sun`, `Rain`, `Fog`, `Snow`, or `Drizzle`, also the U.I. to select the parameters of the Weather was built with Streamlit and the API with Flask. 

## Check-it out the App Deployed in the Streamlit Services

Weather Predictor App Deployed at: https://weather-predictor.streamlit.app/

## Run it Locally

Test it Locally by running the `app.py` file, built with `Streamlit`, and the `api.py` file with `Flask`. Remember first to run the `api.py` file, copy the http url and saved in the API variable of the `app.py` file, and uncomment the code lines.

## App made with Streamlit
```sh
streamlit run app.py
```

## Deployed with Flash
```sh
python3 api.py
```

![](./assets/prev-2.gif)

## Resources
- Weather Predictor Dataset: https://www.kaggle.com/datasets/ananthr1/weather-prediction
