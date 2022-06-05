import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import altair as alt
import random
import difflib

st.title("Career Predictor and Guidance")

#module 1 
data=pd.DataFrame({
    'Jobs'      : ["Enterprise Architect","Full Stack Engineer","Data Scientist","Devops Engineer","Strategy Manager","Machine Learning Engineer","Data Engineer","Software Engineer","Java Developer","Product Manager"],
    'Salary ($)': [125317,107099,116638,113960,130489,140000,120095,120000,101794,144997]})

bar_chart = alt.Chart(data).mark_bar().encode(
        y='Salary ($)',
        x='Jobs',
    )
st.subheader("Click this button to know about the Career trends in 2022")


if st.button('Show Job Trends'):
    st.altair_chart(bar_chart, use_container_width=True)

#module 2

st.subheader("To know which job suits you the most ,type in the answers :")

pipe = joblib.load('pipe1.pkl')
df = joblib.load('df1.pkl')

Logical_quotient_rating = st.selectbox('Logical quotient rating (Rating out of 10)',df['Logical quotient rating'].unique())

coding_skills_rating = st.selectbox('coding skills rating (Rating out of 10)',df['coding skills rating'].unique())

hackathons = st.selectbox('hackathons (ranking out of 10)',df['hackathons'].unique())

public_speaking_points = st.selectbox('public speaking points (Ranking out of 10) ',df['public speaking points'].unique())

self_learning_capability = st.selectbox('self learning capability (Yes:1,No:0)',df['self-learning capability?'].unique())

Extra_courses_did = st.selectbox('Extra courses did (Yes:1,No:0)',df['Extra-courses did'].unique())

Taken_inputs = st.selectbox('Taken inputs (Yes:1,No:0)',df['Taken inputs from seniors or elders'].unique())

worked_in_teams = st.selectbox('Worked in teams (Yes:1,No:0)',df['worked in teams ever?'].unique())

Introvert = st.selectbox('Introvert (Yes:1,No:0)',df['Introvert'].unique())

skills = st.selectbox('Skills (0:Bad,1:ok,2:Good)) ',df['reading and writing skills'].unique())

score = st.selectbox('Memory Score (0:Bad,1:ok,2:Good)',df['memory capability score'].unique())

hard_worker = st.selectbox('Hard worker (Yes:1,No:0)',df['B_hard worker'].unique())

smart_worker = st.selectbox('Smart worker (Yes:1,No:0)',df['B_smart worker'].unique())

Management = st.selectbox('Management (Yes:1,No:0)',df['A_Management'].unique())

Technical = st.selectbox('Technical (Yes:1,No:0)',df['A_Technical'].unique())

if st.button('Predict Career'):
    
    query = np.array([Logical_quotient_rating,coding_skills_rating,hackathons,public_speaking_points,self_learning_capability,Extra_courses_did,Taken_inputs,worked_in_teams,Introvert,skills,score,hard_worker,smart_worker,Management,Technical])
    
    #query =np.array([1,1,1,2,1,1,1,1,1,1,1,1,1,1,1])
    query = query.reshape(1,15)
    
    st.title("The predicted Career is " + pipe.predict(query))

#module 3 
#recommendation system 
st.subheader("Recommendation System")


merged= joblib.load('recommend.pkl')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
vectoriser=TfidfVectorizer()
combined_feature= merged['Course Name']+" "+ merged['University']+" "+merged['Course Description']+" "+merged['Skills']
tf_features=vectoriser.fit_transform(combined_feature)
list_of_all_course_name=merged['Course Name'].tolist()


def recommend(course):
  close_match=difflib.get_close_matches(course,list_of_all_course_name)
  index=merged[merged['Course Name'] == close_match[0] ]['index'].values[0]
  similarity=cosine_similarity(tf_features)
  similarity_score=list(enumerate(similarity[index]))
  sorted_similar_courses=sorted(similarity_score,key = lambda x: x[1] , reverse=True)
  return sorted_similar_courses
  
courses=st.selectbox("List of all the Courses",list_of_all_course_name)
course=st.text_input("Courses you want to search")

if st.button('Show'):
    st.text("Courses Recommanded")
    j=1
    sorted_similar_courses=recommend(course)
    for i in sorted_similar_courses:
        index=i[0]
        title_from_index=merged[merged.index == index]['Course Name'].values[0]
        link_from_index=merged[merged.index == index]['Course URL'].values[0]
        des_from_index=merged[merged.index == index]['University'].values[0]
        if(j < 11):
            st.write(j,'.',title_from_index,"\t\t\t\t\t",link_from_index)
            j+=1