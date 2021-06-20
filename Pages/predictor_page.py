import time

import streamlit as st
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, PassiveAggressiveRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from Pages.page import Page
from config import Config


def predict(data, model):
    model.fit(Config.X_train, Config.y_train.ravel())
    new_data=Config.scaler_x.transform([data])
    res = model.predict(new_data)
    return [res]

class PredictorPage(Page):
    def show_page(self):
            st.write("""# Prediction using Machine Learning Models""")
            st.markdown("""In this page you can insert a new student data and get a predict of
             university rating base on machine learning models.""")

            st.write("""#### Machine Learning Model""")
            machine = st.selectbox("Choose Model:",
                                   ["Linear Regression", "SVR", "Decision Tree", "Random Forest", "Neural Network",
                                    "SGD", "KNN", "Passive Aggressive"])

            st.write("""#### User data""")
            gre = st.number_input("Enter your GRE Score (0 - 340):")
            toefl = st.number_input("Enter Your TOEFL Score (0 - 120):")
            sop = st.slider("Enter Your Statement of Purpose Strength:", 1.0, 5.0, 1.0)
            lor = st.slider("Enter Your Letter of Recomendation Strength:", 1.0, 5.0, 1.0)
            cgpa = st.number_input("Enter Your CGPA (0 - 10):")
            rexp = st.selectbox("Has Research Experience?", ["No", "Yes"])

            if rexp == 'Yes':
                rxp = 1
            else:
                rxp = 0
            if gre < 0 or gre > 340:
                st.error("GRE Score should in the range of 0 - 340")
                st.warning("This may lead to wrong prediction")

            if cgpa < 0 or cgpa > 10.00:
                st.error("CGPA should in the range of 0 - 10.00")
                st.warning("This may lead to wrong prediction")

            if toefl < 0 or toefl > 120.00:
                st.error("TOEFL Score should in the range of 0 - 120")
                st.warning("This may lead to wrong prediction")

            if (toefl <= 120 and toefl >= 0) and (gre <= 340 and gre >= 0) and (cgpa <= 10.00 and cgpa >= 0):

                user_data = [gre, toefl, sop, lor, cgpa, rxp]

                if st.button("Predict"):
                    model = LinearRegression()
                    acc, result = 0, 0
                    if machine == "Linear Regression":
                        model = LinearRegression()
                    if machine == "SVR":
                        model = svm.SVR()
                    if machine == "Decision Tree":
                        model = DecisionTreeRegressor()
                    if machine == "Random Forest":
                        model = RandomForestRegressor(n_estimators=100, max_depth=10)
                    if machine == "Neural Network":
                        model = MLPRegressor(hidden_layer_sizes=(30, 50, 30), activation='relu', solver='adam',
                                             batch_size='auto',
                                             learning_rate='invscaling', learning_rate_init=0.001, shuffle=True)
                    if machine == "SGD":
                        model = SGDRegressor()
                    if machine == "KNN":
                        model = KNeighborsRegressor()
                    if machine == "Passive Aggressive":
                        model = PassiveAggressiveRegressor()

                    result = predict(user_data, model)
                    my_bar = st.progress(0)

                    for percent_complete in range(100):
                        time.sleep(0.1)
                        my_bar.progress(percent_complete + 1)

                    st.write("""### Predicted University Rating on the current profile :""")
                    result = result[0]
                    result = int("%.f" % int(result))

                    print(result)

                    if result >= 4 and result <= 5:
                        st.success(str(result))
                    elif result == 3:
                        st.warning(str(result))
                    else:
                        st.error(str(result))