import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from Pages.page import Page
from config import Config


class FeatureAnalysisPage(Page):
    def show_page(self):
            st.write("""# Exploratory Data Analysis of the Dataset""")

            st.set_option('deprecation.showPyplotGlobalUse', False)
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(Config.admission_df.corr(), annot=True, cmap='Spectral')

            st.pyplot()

            st.markdown("""
                    The features that affect the "Chance to Admit" are: (Importance is in Descending Order)

                    - CGPA (0.87)
                    - GRE Score (0.80)
                    - TOEFL Score (0.79)
                    - University Rating (0.71)
                    - Statement of Purpose (SOP) (0.68)
                    - Letter of Recommendation (LOR) (0.67)
                    - Research Experience (0.55)
                    """
                        )

            st.write("Let's explore these features to get a better understanding")

            select_gr = st.selectbox('Select any Feature',
                                     ['CGPA', 'GRE Score', "TOEFL Score", "University Rating", "Research Experience"])

            if select_gr == 'CGPA':

                st.markdown("""  """)
                st.write("""
                        #### CGPA vs Chance of Admit
                        """)
                st.markdown("""  """)

                st.markdown("""
                        The Cumulative Grade Point Average is a 10 point grading system.
                        From the data shown below, it appears the submissions are normally distributed. With a mean of 8.6 and standard deviation of 0.6.



                        **Moreover, it appears as applicant's CGPA has a strong correlation with their Chance of Admission.**
                        """)

                plt.figure(figsize=(20, 8))
                plt.subplot(1, 2, 1)
                sns.distplot(Config.admission_df['CGPA'], color='r')
                plt.title('CGPA Distribution of Applicants')
                plt.subplot(1, 2, 2)
                sns.regplot(Config.admission_df['CGPA'], Config.admission_df['Chance of Admit'], color='g')
                plt.title('CGPA vs Chance of Admit')

                st.pyplot()

            elif select_gr == 'GRE Score':

                st.markdown("""  """)
                st.write("""
                        #### GRE Score vs Chance of Admit
                        """)
                st.markdown("""  """)

                st.markdown("""
                        The Graduate Record Examination is a standarized exam, often required for admission to graduate and MBA programs globally. It's made up of three components:

                        - Analytical Writing (Scored on a 0-6 scale in half-point increments)
                        - Verbal Reasoning (Scored on a 130-170 scale)
                        - Quantitative Reasoning (Scored on a 130-170 scale)


                        In this dataset, the GRE Score is based on a maximum of 340 points. The mean is 317 with a standard deviation of 11.5.





                        **Moreover, it appears as applicant's GRE Score has a strong correlation with their Chance of Admission, but less than that of CGPA.**
                        """)

                plt.figure(figsize=(20, 8))
                plt.subplot(1, 2, 1)
                sns.distplot(Config.admission_df['GRE Score'], color='m')
                plt.title('Distributed GRE Score of Applicants')

                plt.subplot(1, 2, 2)
                sns.regplot(Config.admission_df['GRE Score'], Config.admission_df['Chance of Admit'], color='b')
                plt.title('GRE Scores vs Chance of Admit')

                st.pyplot()

            elif select_gr == "TOEFL Score":

                st.markdown("""  """)
                st.write("""
                        #### TOEFL Score vs Chance of Admit
                        """)
                st.markdown("""  """)

                st.markdown(
                    """
                    The Test of English as a Foreign Language is a standarized test for non-native English speakers that are choosing to enroll in English-speaking universities.

                    The test is split up into 4 sections:

                    - Reading
                    - Listening
                    - Speaking
                    - Writing


                    All sections are scored out of 30, giving the exam a total score of 120 marks. In this dataset, the TOEFL scores have a mean of 107 and a standard deviation of 6.

                    """
                )

                plt.figure(figsize=(20, 8))
                plt.subplot(1, 2, 1)
                sns.distplot(Config.admission_df['TOEFL Score'], color='g')
                plt.title('Distributed TOEFL Scores of Applicants')

                plt.subplot(1, 2, 2)
                sns.regplot(Config.admission_df['TOEFL Score'], Config.admission_df['Chance of Admit'], color='c')
                plt.title('TOEFL Scores vs Chance of Admit')

                st.pyplot()

            elif select_gr == "University Rating":

                st.markdown("""  """)
                st.write("""
                        #### University Rating vs Number of Applicants
                        """)
                st.markdown("""  """)

                st.markdown(""" 

                        The data analysis shows that the most applicants come from a 3 Star and 2 Star university.

                        """)

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.countplot(Config.admission_df['University Rating'])
                plt.title('University Rating')
                plt.ylabel('Number of Applicants')

                st.pyplot()


            elif select_gr == "Research Experience":
                st.markdown("""  """)
                st.write("""
                        #### Research Experience vs Number of Applicants
                        """)
                st.markdown("""  """)

                st.markdown("""
                        It seems the majority of applicants have research experience. However, this is the least important feature, so it doesn't matter all too much if an applicant has the experience or not.
                        """)

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.countplot(Config.admission_df['Research'])
                plt.title('Research Experience')
                plt.ylabel('Number of Applicants')
                ax.set_xticklabels(['No Research Experience', 'Has Research Experience'])

                st.pyplot()