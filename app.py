import streamlit as st
import pickle
import numpy as np
import real_time_data as rtd

model = pickle.load(open('model.pkl', 'rb'))

temp, hum, wind_vel = rtd.fetch_data()  # Input Location here - Defaults to Kaziranga National Park


def predict_forest(oxygen, humidity, temperature):
    inp = np.array([[oxygen, humidity, temperature]]).astype(np.float64)
    prediction = model.predict_proba(inp)
    pred = '{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)


def main():
    st.title("Streamlit Tutorial")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Forest Fire Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    oxygen = st.text_input("Oxygen", "Type Here")
    humidity = st.text_input("Humidity", "Type Here")
    temperature = st.text_input("Temperature", "Type Here")
    safe_html = """  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> Your forest is safe</h2>
       </div>
    """
    danger_html = """  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> Your forest is in danger</h2>
       </div>
    """

    if st.button("Predict"):
        output = predict_forest(oxygen, humidity, temperature)
        st.success('The probability of fire taking place is {}'.format(output))

        if output > 0.5:
            st.markdown(danger_html, unsafe_allow_html=True)
        else:
            st.markdown(safe_html, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
