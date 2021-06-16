import streamlit as st

from Pages.page import Page
from config import Config


class IntroductionPage(Page):
    def show_page(self):
        st.write("""# A Comparison of Regression Models for Predicting Graduate Admission""")
        st.write("""### Introduction""")
        st.markdown("Prospective graduate students always face a dilemma deciding universities"
                    " of their choice while applying to master’s programs. While there are a good number "
                    "of predictors and consultancies that guide a student, they aren’t always reliable since"
                    " decision is made on the basis of select past admissions. In this website, we present a"
                    " Machine Learning based method where we compare different regression algorithms, such as"
                    " Linear Regression, Support Vector Regression, Decision Trees, Random Forest,  Neural "
                    "Network, SGD, KNN Regression and Passive Aggressive. given the "
                    "profile of the student. We then compute error functions for the different models and"
                    " compare their performance to select the best performing model.")

        st.write("""### Dataset""")
        st.dataframe(Config.admission_df)
        st.markdown(
            "The dataset, we are using, consists of 400 grad student’s records containing information about the GRE Scores (in 340 scale), TOEFL Scores (in 120 scale), University Rating (on scale of 5), Statement of Purpose Strength (on scale of 5), Letter of Recommendation Strength (on scale of 5), CGPA (Undergraduate Grade Point), Research Experience (yes or no in which 1 denotes Yes and 0 denotes No) and Chance of Admit (a value between 0 and 1), being the target variable. In the data-set, 6 features are continuous with only 1 feature, Research Experience as categorical.")

        st.write("""#### More information about our dataset""")
        st.write(Config.admission_df.describe())