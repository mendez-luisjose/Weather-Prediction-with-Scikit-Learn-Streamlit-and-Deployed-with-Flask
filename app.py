#App made with Streamlit

#Loading Modules Needed
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import plost
import pickle

#URL of the API made with Flask
API = "http://127.0.0.1:5000"

MODEL_PATH = f'./model/weather_model.pkl'
SCALER_PATH = f'./model/scaler.pkl'

IMG_SIDEBAR_PATH = "./assets/img.jpg"

#Function to load the Model and the Scaler
def load_pkl(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj

model = load_pkl(MODEL_PATH)
scaler = load_pkl(SCALER_PATH)

#Function to load the Iris Dataset
def get_clean_data():
  data = pd.read_csv("./dataset/weather_dataset.csv")

  return data

#Sidebar of the Streamlit App
def add_sidebar():
  st.sidebar.header("Weather Predictor `App ⛈️`")
  image = np.array(Image.open(IMG_SIDEBAR_PATH))
  st.sidebar.image(image)
  st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
  st.sidebar.write("This Artificial Intelligence App can Predicts the Future Weather Given their Corresponding Parameters.")

  st.sidebar.subheader('Select the Weather Parameters ✅:')
  
  data = get_clean_data()
  
  slider_labels = [
        ("Precipitation", "precipitation"),
        ("Max Temperature", "temp_max"),
        ("Min Temperature", "temp_min"),
        ("Wind", "wind"),
    ]

  input_dict = {}

  for label, key in slider_labels:
    input_dict[key] = st.sidebar.slider(
      label,
      min_value=float(0),
      max_value=float(data[key].max()),
      value=float(data[key].mean())
    )

  st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
  st.sidebar.markdown('''
  🧑🏻‍💻 Created by [Luis Jose Mendez](https://github.com/mendez-luisjose).
  ''')

  return input_dict

def get_scaled_values(input_dict):
  data = get_clean_data()
  
  X = data.drop(['weather'], axis=1)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict

#Radar Chart Function
def get_radar_chart(input_data):
  input_data = get_scaled_values(input_data)
  
  categories = ['Precipitation', 'Max Temperature', 'Min Temperature', 'Wind']

  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['precipitation'], input_data['temp_max'], input_data['temp_min'],
          input_data['wind']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
  ))
  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
  )
  
  return fig

#Receiving Prediction Results from the API
def add_predictions(input_data) :
    input_array = np.array(list(input_data.values())).reshape(1, -1).tolist()

    input_array_scaled = scaler.transform(input_array)
    pred_result = model.predict(input_array_scaled)

    pred_result = int(pred_result[0])
    prob_drizzle = round(model.predict_proba(input_array_scaled)[0][0], 2)
    prob_rain = round(model.predict_proba(input_array_scaled)[0][1], 2)
    prob_sun = round(model.predict_proba(input_array_scaled)[0][2], 2)
    prob_snow = round(model.predict_proba(input_array_scaled)[0][3], 2)
    prob_fog = round(model.predict_proba(input_array_scaled)[0][4], 2)

    #Run first the api.py file and the paste the URL in the API Variable if you want to deploy the Model with Flask and uncomment the next lines
    #data = {'array': input_array}

    #resp = requests.post(API, json=data)
    
    #pred_result = resp.json()["Results"]["result"]
    #prob_drizzle = resp.json()["Results"]["prob_drizzle"]
    #prob_rain = resp.json()["Results"]["prob_rain"]
    #prob_sun = resp.json()["Results"]["prob_sun"]
    #prob_snow = resp.json()["Results"]["prob_snow"]
    #prob_fog = resp.json()["Results"]["prob_fog"]

    st.markdown("### Weather Prediction ✅")
    st.write("<span class='diagnosis-label'>Machine Learning Model Result:</span>",  unsafe_allow_html=True)
    
    if pred_result == 0:
      st.write("<span class='diagnosis drizzle'>Drizzle</span>", unsafe_allow_html=True)
    elif pred_result == 1 :
      st.write("<span class='diagnosis rain'>Rain</span>", unsafe_allow_html=True)
    elif pred_result == 2:
      st.write("<span class='diagnosis sun'>Sun</span>", unsafe_allow_html=True)
    elif pred_result == 3 :
      st.write("<span class='diagnosis snow'>Snow</span>", unsafe_allow_html=True)
    elif pred_result == 4 :
      st.write("<span class='diagnosis fog'>Fog</span>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1 :
        st.metric("Probability:", f"{prob_drizzle}%", "Drizzle")
    
    with col2:
        st.metric("Probability:", f"{prob_rain}%", "Rain")
      
    with col3: 
        st.metric("Probability:", f"{prob_sun}%", "Sun")

    col4, col5 = st.columns([1, 1])
    with col4 :
        st.metric("Probability:", f"{prob_snow}%", "Snow")
    
    with col5:
        st.metric("Probability:", f"{prob_fog}%", "Fog")

    st.write("`This Artificial Intelligence can Assist for any Scientific about the Upcoming Weather, but Should Not be used as a Substitute for a Final Diagnosis and Prediction.`")
    

def main() :  
    st.set_page_config(
        page_title="Weather Predictor",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
  
    input_data = add_sidebar()

    st.markdown(
        """
        <style>
        [data-testid="stSidebar][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }

        </style>
        """,
        unsafe_allow_html=True
    )    

    with st.container() :
        st.title("Weather Predictor ⛅️")
        st.write("This App predicts using a KNeighborsClassifier Machine Learning Model whether a given parameters the Upcoming Weather is eather Drizzle, Sun, Snow, Fog or Rain. You can also Update the measurements by hand using sliders in the sidebar.")
        st.markdown("<hr/>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])

    df = pd.read_csv("./assets/weather_classes.csv")

    with col1:
        st.markdown('### Radar Chart of the Parameters 📊')
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

        st.markdown('### Bar Chart of the Weather Classes 📉')
        st.markdown("---", unsafe_allow_html=True)

        plost.bar_chart(
            data=df,
            bar='Weather',
            value='Number of that Class', 
            legend='bottom',
            use_container_width=True,
            color='Weather')        
        

    with col2:
        st.markdown('### Donut Chart of the Weather Classes 📈')

        plost.donut_chart(
            data=df,
            theta="Number of that Class",
            color='Weather',
            legend='bottom', 
            use_container_width=True)
        
        st.markdown("<hr/>", unsafe_allow_html=True)
        add_predictions(input_data)

if __name__ == "__main__" :
    main()

    print("App Running!")