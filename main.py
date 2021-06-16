import streamlit as st

from Pages.feature_analysis_page import FeatureAnalysisPage
from Pages.introduction_page import IntroductionPage
from Pages.models_analysis_page import ModelsAnalysisPage
from Pages.predictor_page import PredictorPage

if __name__ == '__main__':
    st.sidebar.title("Graduate Admission Prediction")
    menu = st.sidebar.radio('Navigation', ('Introduction', "Feature Analysis", "Models Analysis", "Predictor"))
    st.sidebar.title("Details")
    st.sidebar.info(
        "Author: Zvi Berger and Liel Shuker")
    st.sidebar.info(
        "This Project is based on the paper - 'A Comparison of Regression Models for Predicting Graduate Admission'")
    st.sidebar.info(
        "[The paper](https://drive.google.com/file/d/17su4WNKIwrOA5WUXXS3qIPStu3eLfEfv/view?usp=sharing)")
    st.sidebar.info(
        "[Kaggle Dataset](https://www.kaggle.com/mohansacharya/datasets)")
    st.sidebar.info("[Presentation]()")
    st.sidebar.info("[Github](https://github.com/BergerZvika/Graduate-Admission-Prediction)")

    introduction = IntroductionPage()
    feature_analysis = FeatureAnalysisPage()
    predictor = PredictorPage()
    models_analysis = ModelsAnalysisPage()

    if menu == 'Introduction':
        introduction.show_page()

    if menu == 'Feature Analysis':
        feature_analysis.show_page()

    if menu == 'Predictor':
        predictor.show_page()

    if menu == "Models Analysis":
        models_analysis.show_page()


