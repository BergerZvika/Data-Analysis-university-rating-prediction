import streamlit as st

from Pages.page import Page
from config import Config


class IntroductionPage(Page):
    def show_page(self):
        st.write("""# A Comparison of Regression Models for Predicting Graduate Admission""")
        st.write("""### Introduction""")
        st.markdown("This project research about ")

        st.write("""### Dataset""")
        st.dataframe(Config.admission_df)
        st.markdown(
            "The dataset, we are using, consists of 400 grad student’s records containing information about the GRE Scores (in 340 scale), TOEFL Scores (in 120 scale), University Rating (on scale of 5), Statement of Purpose Strength (on scale of 5), Letter of Recommendation Strength (on scale of 5), CGPA (Undergraduate Grade Point), Research Experience (yes or no in which 1 denotes Yes and 0 denotes No) and Chance of Admit (a value between 0 and 1), being the target variable. In the data-set, 6 features are continuous with only 1 feature, Research Experience as categorical.")

        st.write("""#### More information about our dataset""")
        st.write(Config.admission_df.describe())
