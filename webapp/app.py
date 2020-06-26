import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from sklearn.externals import joblib
# Loading Our final trained Knn model
model = open("../model/LR_l1_tuned.mod", "rb")
clf = pickle.load(model)
# scaler = joblib.load('../model/LR_l1_tuned.scaler')
st.title("Cancer Classification App")

st.sidebar.title("Features")
# Intializing
parameter_list = list(np.genfromtxt('../data/raw/breast-cancer-wisconsin.txt',
                                    delimiter=',', max_rows=1, comments=None, dtype='str'))[2:-1]
parameter_input_values = []
parameter_default_values = ['5'] * len(parameter_list)
values = []

# Display
for parameter, parameter_df in zip(parameter_list, parameter_default_values):

    values = st.sidebar.slider(label=parameter, key=parameter, value=float(
        parameter_df), min_value=0.0, max_value=10.0, step=0.1)
    parameter_input_values.append(values)

input_variables = pd.DataFrame(
    [parameter_input_values], columns=parameter_list, dtype=float)
st.write('\n\n')

# input_variables = scaler.transform(input_variables)
# st.write(input_variables.shape)


if st.button("Click Here to Classify"):
    prediction = clf.predict(input_variables)
    if prediction == 0:
        st.write('Benign')
    elif prediction == 1:
        st.write(
            'Malignant')
