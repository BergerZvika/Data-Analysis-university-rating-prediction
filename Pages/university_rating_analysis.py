import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from Pages.page import Page
from config import Config
import pandas as pd
import seaborn as sns



class UniversityRatingAnalysisPage(Page):
    def show_page(self):
        st.write("""# Data Analysis of the University Rating""")
        st.markdown("""In this page we analysis the data to answer our main question: Does student choose university with rating suitable for them? """)
        st.write("""## Data""")
        st.markdown("""We sort the data by CGPA values and then split the data to 3 equals group. The first group include
                    records with the least value. The second group include records with means values, Andthe last group
                     include records with the best value.""")

        data = Config.admission_df.sort_values(by=['CGPA'],inplace=False)
        data_split = np.array_split(data, 3)
        weak = data_split[0]
        medium = data_split[1]
        strong = data_split[2]
        st.markdown("""### Show the data groups:""")
        st.markdown("""Weak group:""")
        st.dataframe(weak)
        st.markdown("""Mean group:""")
        st.dataframe(medium)
        st.markdown("""Best group:""")
        st.dataframe(strong)

        st.write("""## Analysis""")
        st.markdown("""### Show University rating of each group:""")
        st.markdown("""Weak group:""")

        plt.figure(figsize=(20, 8))
        plt.subplot(1, 2, 1)
        sns.distplot(weak['University Rating'], color='g')
        plt.title('Distributed University Rating of weak group')
        st.pyplot()

        st.markdown("""Mean group:""")

        plt.figure(figsize=(20, 8))
        plt.subplot(1, 2, 1)
        sns.distplot(medium['University Rating'], color='g')
        plt.title('Distributed University Rating of medium group')
        st.pyplot()

        st.markdown("""Best group:""")

        plt.figure(figsize=(20, 8))
        plt.subplot(1, 2, 1)
        sns.distplot(strong['University Rating'], color='g')
        plt.title('Distributed University Rating of strong group')
        st.pyplot()

