#!/usr/bin/env python
# coding: utf-8

# In[17]:


import streamlit as st
import numpy as np
import tensorflow as tf
import joblib


# In[18]:


#model=tf.keras.models.load_model("student_model_x.h5")
model = tf.keras.models.load_model("student_model_x.h5", compile=False)


# In[19]:
st.title("🎓 Student Performance Predictor")


scalar=joblib.load("scaler_student_performance.pkl")



# In[20]:





# In[51]:


## User Inputs
Hours_Studied=st.slider("hours Studied",0,44,6)
Attendance=st.slider("Attendance",10,100,60)
Sleep_Hours=st.slider("Sleep Hours",0,12,8)
Parental_Education_Level=st.selectbox("Parent Education Level 0=College,1=High school, 2=Post Gratuate",[0,1,2])
Previous_Scores=st.slider("Previous year Percentage",0,100,70)
Gender=st.selectbox("Gender 0-Female, 1-Male",[0,1])
Motivation_Level=st.selectbox("Motivation Level:  0=High,1=Low, 2=Medium",[0,1,2])
Extracurricular_Activities=st.selectbox("Extra Curricular 0=No, 1=YES",[0,1])


# In[52]:


import numpy as np

input_data = np.array([[
    Hours_Studied,
    Attendance,
    Sleep_Hours,
    Parental_Education_Level,
    Previous_Scores,
    Gender,
    Motivation_Level,
    Extracurricular_Activities
]], dtype=float)



# In[53]:


input_scaled=scalar.transform(input_data)


# In[54]:


## 


# In[55]:


if st.button("Pridict Score"):
    prediction=model.predict(input_scaled)
    st.success(f'Pridected Exam Score: {prediction[0][0]:.2f}')


# In[ ]:




