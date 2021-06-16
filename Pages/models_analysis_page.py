import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, PassiveAggressiveRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from Pages.page import Page
from config import Config


def evaluation(model):
    model.fit(Config.X_train, Config.y_train.ravel())
    y_predict = model.predict(Config.X_test)
    mse = mean_squared_error(Config.y_test, y_predict)
    r2 = r2_score(Config.y_test, y_predict)
    return [mse, r2, [Config.y_test, y_predict]]

class ModelsAnalysisPage(Page):
    def show_page(self):
            st.write("""# Data Analysis of the Models""")

            st.markdown("""This page compare between machine learning models. We show the MSE and R-Square values for each model.
                        Below the table we put some graph to show the diffrent off the predicted values on models.""")

            st.markdown("""  """)
            st.markdown("""## Table""")

            mse_linear, r2_linear, grp_linear = evaluation(LinearRegression())
            mse_svr, r2_svr, grp_svr = evaluation(svm.SVR())
            mse_tree, r2_tree, grp_tree = evaluation(DecisionTreeRegressor())
            mse_forest, r2_forest, grp_forest = evaluation(RandomForestRegressor(n_estimators=100, max_depth=10))
            mse_nn, r2_nn, grp_nn = evaluation(MLPRegressor(hidden_layer_sizes=(30, 50, 30), activation='relu'
                                                            , solver='adam', batch_size='auto',
                                                            learning_rate='invscaling',
                                                            learning_rate_init=0.001, shuffle=True))
            mse_sgd, r2_sgd, grp_sgd = evaluation(SGDRegressor())
            mse_knn, r2_knn, grp_knn = evaluation(KNeighborsRegressor())
            mse_pa, r2_pa, grp_pa = evaluation(PassiveAggressiveRegressor())

            r2 = [r2_linear, r2_svr, r2_tree, r2_forest, r2_nn, r2_sgd, r2_knn, r2_pa]
            mse = [mse_linear, mse_svr, mse_tree, mse_forest, mse_nn, mse_sgd, mse_knn, mse_pa]

            acc = []
            for r in r2:
                acc.append(f'{int(r * 100)}%')

            data = {'Models': ['Linear Regression', 'SVR', "Decision Tree",
                               'Random Forest Regression', "Neural Network", "SGD",
                               "KNN Regression", "Passive Aggressive"],
                    'R-Square Score': r2,
                    'Mean Sqaure Error': mse,
                    'Accuracy': acc
                    }
            table = pd.DataFrame(data, columns=['Models', 'R-Square Score', 'Mean Sqaure Error', 'Accuracy'])

            st.table(table)

            # graph
            st.markdown("""## Graph""")

            plt.plot(grp_linear[0], grp_linear[1], "^", color='r')
            plt.ylabel("Predicted Chance of Admission")
            plt.xlabel("Actual Chance of Admission")
            plt.title('Linear Regression')
            st.pyplot()
            st.markdown("""""")

            plt.plot(grp_svr[0], grp_svr[1], "^", color='r')
            plt.ylabel("Predicted Chance of Admission")
            plt.xlabel("Actual Chance of Admission")
            plt.title('Support Vector Regression')
            st.pyplot()
            st.markdown("""""")

            plt.plot(grp_tree[0], grp_tree[1], "^", color='r')
            plt.ylabel("Predicted Chance of Admission")
            plt.xlabel("Actual Chance of Admission")
            plt.title('Decision Tree')
            st.pyplot()
            st.markdown("""""")

            plt.plot(grp_forest[0], grp_forest[1], "^", color='m')
            plt.ylabel("Predicted Chance of Admission")
            plt.xlabel("Actual Chance of Admission")
            plt.title('Random Forest Regression')
            st.pyplot()
            st.markdown("""""")

            plt.plot(grp_nn[0], grp_nn[1], "^", color='m')
            plt.ylabel("Predicted Chance of Admission")
            plt.xlabel("Actual Chance of Admission")
            plt.title('Neural Network')
            st.pyplot()
            st.markdown("""""")

            plt.plot(grp_sgd[0], grp_sgd[1], "^", color='m')
            plt.ylabel("Predicted Chance of Admission")
            plt.xlabel("Actual Chance of Admission")
            plt.title('SGD')
            st.pyplot()
            st.markdown("""""")

            plt.plot(grp_knn[0], grp_knn[1], "^", color='m')
            plt.ylabel("Predicted Chance of Admission")
            plt.xlabel("Actual Chance of Admission")
            plt.title('KNN Regression')
            st.pyplot()
            st.markdown("""""")

            plt.plot(grp_pa[0], grp_pa[1], "^", color='m')
            plt.ylabel("Predicted Chance of Admission")
            plt.xlabel("Actual Chance of Admission")
            plt.title('Passive Aggressive')
            st.pyplot()
            st.markdown("""""")

