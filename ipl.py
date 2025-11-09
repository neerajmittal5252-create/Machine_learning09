import streamlit as st
import pickle 
import numpy as np
import pandas as pd
import sklearn

with open("pipe.pkl", "rb") as file:
    model = pickle.load(file)

st.title('IPL Win Predictor')
teams=['Sunrisers Hyderabad',
            'Mumbai Indians',
            'Royal Challengers Bangalore',
            'Kolkata Knight Riders',
            'Kings XI Punjab',
            'Chennai Super Kings',
            'Rajasthan Royals',
            'Delhi Capitals']

cities=['Port Elizabeth', 'Bengaluru', 'Pune', 'Bangalore', 'Chennai',
       'Delhi', 'Mumbai', 'Ahmedabad', 'Chandigarh', 'Jaipur', 'Kolkata',
       'Kimberley', 'Abu Dhabi', 'Mohali', 'Centurion', 'Johannesburg',
       'Durban', 'Ranchi', 'Nagpur', 'Hyderabad', 'Visakhapatnam',
       'Bloemfontein', 'East London', 'Cuttack', 'Indore', 'Raipur',
       'Sharjah', 'Cape Town', 'Dharamsala']

batting_team=st.selectbox("Select batting team",sorted(teams))

bowling_team=st.selectbox("Select bowling team",sorted(teams))

cs=st.number_input("Current Score",min_value=0)
wo=st.number_input("Wickets Out",max_value=9,min_value=0)
oc=st.number_input("Overs Completed",max_value=20,min_value=0)
tar=st.number_input("Target")
city=st.selectbox("Select venue city",sorted(cities))

if(tar < cs):
    st.success(batting_team + "- " + str(round(100)) + "%")
    st.success(bowling_team + "- " + str(round(0)) + "%")
else:
    if st.button("Predict Probability"):
        runs_left=tar-cs   
        balls_left=120-(oc*6)
        wickets=10-wo
        crr=cs/oc
        rrr=(runs_left*6)/balls_left

        input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[tar],'crr':[crr],'rrr':[rrr]})
        result = model.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]
        st.success(batting_team + "- " + str(round(win*100)) + "%")
        st.success(bowling_team + "- " + str(round(loss*100)) + "%")