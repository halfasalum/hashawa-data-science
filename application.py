import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
from datetime import datetime
from sqlalchemy import create_engine, text
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import altair as alt

user = 'root'
password = 'root'
host = 'localhost'
port = '8889'
database = 'hashawa2'
# Create SQLAlchemy engine
engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}')   

st.set_page_config(layout="wide")
st.balloons()


def age_range(age):
    if (age > 60):
        return '60+'
    elif (age > 50):
        return '51 - 60'
    elif (age > 40):
        return '41 - 50'
    elif (age > 30):
        return '31 - 40'
    elif (age > 25):
        return '26 - 30'
    else:
        return '18 - 25'


def dataPreprocessing():
        #progress_bar = st.progress(0)
    with engine.connect() as conn:
        customersId = conn.execute("SELECT customer_id as id from CUSTOMERS WHERE income > 0 AND income <= 7000000")
        customersId = pd.DataFrame(customersId)
        customersId.to_csv('customers_insight.csv', index=False)

        customers = pd.read_csv("customers_insight.csv")
        customers['success_loan']   = 0
        customers['failed_loans']   = 0
        customers['income']         = 0
        customers.to_csv('customers_insight.csv', index=False)
        customers = pd.read_csv("customers_insight.csv")
        total_rows = len(customers)
        for index, row in customers.iterrows():
            customer_id = row['id']
            query = text("SELECT COUNT(*) FROM LOANS WHERE customer_id = :customer_id  AND loan_status = 5")
            loansSuccess = conn.execute(query,customer_id=customer_id).scalar()
            #st.write(loansSuccess)
            customers.at[index,'success_loan'] = loansSuccess
            query = text("SELECT COUNT(*) FROM LOANS WHERE customer_id = :customer_id  AND loan_status = 6")
            loanFailed   = conn.execute(query,customer_id=customer_id).scalar()
            customers.at[index,'failed_loans'] = loanFailed
            query = text("SELECT income FROM CUSTOMERS WHERE customer_id = :customer_id")
            income   = conn.execute(query,customer_id=customer_id).scalar()
            customers.at[index,'income'] = income
            customers.to_csv('customers_insight.csv', index=False)
            #progress_bar.progress((index + 1) / total_rows)

    with engine.connect() as conn:
        result = conn.execute("""
                              SELECT 
                              customer_id, 
                              CONCAT(fname,' ', mname, ' ',lname) AS name, 
                              income, 
                              age as dob,
                              CUSTOMERS.gender as gender,
                              income
                              FROM CUSTOMERS INNER JOIN tbl_customers ON tbl_customers.customerId = CUSTOMERS.customer_id
                              WHERE income > 0 AND income <= 7000000
                              """)
        
        customers = pd.DataFrame(result)
        customers.to_csv('customers.csv', index=False)


#dataPreprocessing()

loans = pd.read_csv("loans.csv")
customers   = pd.read_csv("customers_insight.csv")
loan_by_age = pd.read_csv("df_mean_loan_by_age_range.csv")
loan_by_male = pd.read_csv("df_number_loan_per_male_per_year.csv")
loan_by_female = pd.read_csv("df_number_loan_per_female_per_year.csv")


def customerAnalyzer():
    
    #progress_bar = st.progress(0)
    with engine.connect() as conn:

        customers = pd.read_csv("customers_insight.csv")
        customers['success_loan']   = 0
        customers['failed_loans']   = 0
        customers['income']         = 0
        customers.to_csv('customers_insight.csv', index=False)
        customers = pd.read_csv("customers_insight.csv")
        total_rows = len(customers)
        for index, row in customers.iterrows():
            customer_id = row['id']
            query = text("SELECT COUNT(*) FROM LOANS WHERE customer_id = :customer_id  AND loan_status = 5")
            loansSuccess = conn.execute(query,customer_id=customer_id).scalar()
            #st.write(loansSuccess)
            customers.at[index,'success_loan'] = loansSuccess
            query = text("SELECT COUNT(*) FROM LOANS WHERE customer_id = :customer_id  AND loan_status = 6")
            loanFailed   = conn.execute(query,customer_id=customer_id).scalar()
            customers.at[index,'failed_loans'] = loanFailed
            query = text("SELECT income FROM CUSTOMERS WHERE customer_id = :customer_id")
            income   = conn.execute(query,customer_id=customer_id).scalar()
            customers.at[index,'income'] = income
            customers.to_csv('customers_insight.csv', index=False)
            #progress_bar.progress((index + 1) / total_rows)

#customerAnalyzer()

def nomalizeData():
    customers   = pd.read_csv("customers_insight.csv")
    customers   = customers[customers['success_loan'] > 0]
    customers.to_csv("customers_insight.csv", index=False)
    customers   = customers[customers['income'] < 7000000]
    customers.to_csv("customers_insight.csv", index=False)
    customers   = customers[customers['income'] > 0]
    customers.to_csv("customers_insight.csv", index=False)

    scale = MinMaxScaler()
    customers[['nomalized_success_loan','nomalized_failed_loans','nomalized_income']] = scale.fit_transform(customers[['success_loan','failed_loans','income']])
    kMeans = KMeans(n_clusters=3,random_state=42)
    customers['cluster'] = kMeans.fit_predict(customers[['nomalized_success_loan','nomalized_failed_loans','nomalized_income']])
    cluster_centers = kMeans.cluster_centers_
    st.write(cluster_centers)
    customers['priority'] = customers['cluster'].map({0: 'Low Priority', 1: 'Medium Priority', 2: 'High Priority'})
    st.table(customers.head())
    customers = customers.dropna()
    customers['nomalized_success_loan'] = pd.to_numeric(customers['nomalized_success_loan'], errors='coerce')
    customers['nomalized_failed_loans'] = pd.to_numeric(customers['nomalized_failed_loans'], errors='coerce')
    customers['nomalized_income'] = pd.to_numeric(customers['nomalized_income'], errors='coerce')
    customers['cluster'] = pd.to_numeric(customers['cluster'], errors='coerce')

    X = [['nomalized_success_loan','nomalized_failed_loans','nomalized_income']]
    y = customers['cluster']
    model = LinearRegression()
    model.fit(X,y)
    alpha, beta, gamma = model.coef_
    intercept = model.intercept_
    st.write("(success loan) = {alpha}")
    st.write("(failed loans) = {beta}")
    st.write("(income) = {gamma}")
    st.write("Intercept = {intercept}")

#nomalizeData()



def nomalizeData2():
    # Load and clean data
    customers = pd.read_csv("customers_insight.csv")
    customers = customers[customers['success_loan'] > 0]
    customers = customers[customers['income'] < 7000000]
    customers = customers[customers['income'] > 0]

    # Normalize data
    scale = MinMaxScaler()
    customers[['nomalized_success_loan', 'nomalized_failed_loans', 'nomalized_income']] = scale.fit_transform(
        customers[['success_loan', 'failed_loans', 'income']]
    )

    # K-Means Clustering
    kMeans = KMeans(n_clusters=3, random_state=42)
    customers['cluster'] = kMeans.fit_predict(
        customers[['nomalized_success_loan', 'nomalized_failed_loans', 'nomalized_income']]
    )

    # Map clusters to priority
    customers['priority'] = customers['cluster'].map({0: 'Low Priority', 1: 'Medium Priority', 2: 'High Priority'})
    #st.table(customers.head())

    # Drop rows with NaN values
    customers = customers.dropna()

    # Features and Target for Linear Regression
    X = customers[['nomalized_success_loan', 'nomalized_failed_loans', 'nomalized_income']]
    y = customers['cluster']

    # Fit Linear Regression Model
    model = LinearRegression()
    model.fit(X, y)

    # Extract coefficients and intercept
    alpha, beta, gamma = model.coef_
    intercept = model.intercept_

    # Display results
    #st.write(f"Alpha (success loan) = {alpha}")
    #st.write(f"Beta (failed loans) = {beta}")
    #st.write(f"Gamma (income) = {gamma}")
    #st.write(f"Intercept = {intercept}")
    customers.to_csv("customers_insight.csv", index=False)
    customers = pd.read_csv("customers_insight.csv")
    st.divider()
    #st.text('Final Customer Score')
    customers['score']  =  (alpha * customers['nomalized_success_loan'] +
                      beta * customers['nomalized_failed_loans'] +
                      gamma * customers['nomalized_income'] +
                      intercept)
    #st.write(customers['score'].min())
    #st.write(customers['score'].max())
    #st.scatter_chart(customers['score'])

    customers['final_priority'] = customers['score'].apply(assign_priority)
    customers.to_csv("customers_insight.csv", index=False)

    #st.table(customers.head())

def assign_priority(score):
    thresholds = {
    'High Priority': score > 2.00,
    'Medium Priority': score > 1.00
    }
    if score >= 2.00:
        return 'High Priority'
    elif score >= 1.00:
        return 'Medium Priority'
    else:
        return 'Low Priority'

# Call the function


def customerScore(alpha,beta,gamma, intercept):
    customers           = pd.read_csv("customers_insight.csv")
    customers['score']  =  (alpha * customers['nomalized_success_loan'] +
                      beta * customers['nomalized_failed_loans'] +
                      gamma * customers['nomalized_income'] +
                      intercept)
    st.table(customers.head())

def scaleData():
    st.write('Scalling data')



st.header("Loan Analysis and Prediction System")

home, insight, recomendation, loanPrediction, customerSection, graph = st.tabs(['Home','Data Insight','Loan Recomendation','Loan Prediction','Customer Analaysis','Graphs'])

with home:
    st.subheader("Introduction")
    st.text('This project aims to develop a comprehensive loan analysis and prediction system for a credit company. Baseline [Tanzania, Dar es salaam]')
    st.text('The system leverages machine learning techniques to analyze customer datasets, provide loan approval recommendations, predict loan amounts, and categorize customers based on their behaviors.')
    st.text('The project focuses on real-time data analysis using a streaming approach in Python to ensure timely and accurate decision-making.')
    st.subheader("Objectives")
    st.text("Dataset Analysis: Analyze customer information, loan history, and payment history to identify patterns and insights.")
    st.text("Loan Approval Recommendation: Develop a model to recommend whether a loan should be approved based on key customer attributes.")
    st.text("Loan Amount Prediction: Predict the optimal loan amount for approved customers using predictive modeling techniques.")
    st.text("Customer Categorization: Categorize customers into different risk categories based on their credit profile and payment behavior.")
    st.subheader("Conclusion")
    st.text("This project will enable the credit company to make data-driven decisions, improve loan approval accuracy, optimize loan amounts, and better understand customer behavior.")
    st.text("The real-time processing capability will ensure timely responses to customer applications and enhance overall operational efficiency.")


with recomendation:
    st.subheader('Loan Approval Recomendation')
    st.text("This section will provide the recommndation for loan approval based on the inpt provided")
    with st.form(key='loanApproval'):
        gender = st.selectbox(
            'Please choose customer gender',
            ("Male","Female"),
            index=None
            )
        dob = st.date_input(
            "Customer Birthdate",
            format="YYYY-MM-DD",
            value=None
        )
        marital = st.selectbox(
            'Please choose customer marital status',
            ("Married","Single"),
            index=None
            )
        income = st.number_input(
            "Customer income",
            min_value = 10000,
            value = None,
            step = None
        )
        loan = st.number_input(
            "Customer Loan Requested",
            min_value = 50000,
            value = None,
            step = None
        )
        submit = st.form_submit_button(label='Process')
        if(submit):
            recomendationModel = joblib.load("loan_default_model.joblib")
            if(gender == 'Male'):
                gender = 1
            else:
                gender = 0

            if(marital == 'Married'):
                marital = 1
            else:
                marital = 0
            
            dob = pd.to_datetime(dob)
            today = datetime.today()
            age = (today.year - dob.year)
            ratio = loan/income
            customer_data = np.array([[gender,age,marital,income,loan,ratio]])
            recomend = recomendationModel.predict(customer_data).astype(int)
            if (age > 18):
                if (recomend == 5):
                    st.write("Recommended For Approval")
                    predictionModel = joblib.load("loan_amount_prediction_model.joblib")
                    customer_data = np.array([[gender,age,income,marital]]) 
                    predict_loan = predictionModel.predict(customer_data).astype(int)
                    st.write(f"Proposed Loan  For Customer Maximum To : {predict_loan}")

                else:
                    st.write("NOT Recommended For Approval")
            else:
                st.write("NOT Recommended For Approval")

with loanPrediction:
    st.subheader('Loan Amount Recomendation')
    with st.form(key='loanAmountForm'):
        gender = st.selectbox(
            'Please choose customer gender',
            ("Male","Female"),
            index=None
            )
        dob = st.date_input(
            "Customer Birthdate",
            format="YYYY-MM-DD",
            value=None
        )
        marital = st.selectbox(
            'Please choose customer marital status',
            ("Married","Single"),
            index=None
            )
        income = st.number_input(
            "Customer income",
            min_value = 10000,
            value = None,
            step = None
        )
        loan = st.number_input(
            "Customer Loan Requested",
            min_value = 50000,
            value = None,
            step = None
        )
        submit = st.form_submit_button(label='Process')
    
    if(submit):
        predictionModel = joblib.load("loan_amount_prediction_model.joblib")
        if(gender == 'Male'):
            gender = 1
        else:
            gender = 0

        if(marital == 'Married'):
            marital = 1
        else:
            marital = 0
        dob = pd.to_datetime(dob)
        today = datetime.today()
        age = (today.year - dob.year)
        ratio = loan/income
        customer_data = np.array([[gender,age,income,marital]]) 
        predict_loan = predictionModel.predict(customer_data).astype(int)
        if(age > 18):
            st.write(f"Proposed Loan  For Customer To Maximum Of : {predict_loan}")
        else:
            st.write("NOT recommended for loan")
        #predict_loan = locale.currency(predict_loan, grouping=True)
        

with customerSection:
    st.subheader('Customer Analysis')
    customers = pd.read_csv('customers.csv')
    with st.form(key = 'custoomerAnalysis'):
        name = st.selectbox(
            "Select customer",
            customers['name'],
            index=None
        )
        submit = st.form_submit_button("Analyse")
    st.divider()
    st.text('Analysis report')
    if(submit):
        customers_insight   = pd.read_csv("customers_insight.csv")
        customer_data       = customers.loc[customers['name'] == name] 
        data = customers_insight.loc[customers_insight['id'] == customer_data['customer_id'].values[0]]
        st.write(data) 



with graph:
    st.text('Dataset Graphs')



def highPriorityCustomers():
    customers       = pd.read_csv("customers.csv")
    st.table(customers.head())

 
    
def loansByAge():
    customers   = pd.read_csv("customers.csv")
    customers['dob']    = pd.to_datetime(customers['dob'])
    today               = datetime.today()
    customers['age']    = (today - customers['dob']).astype('<m8[Y]').astype(int)
    customers['year_range']     = customers['age'].apply(age_range)
    customers.to_csv("customers.csv", index=False)
#loansByAge()

customers               = pd.read_csv("customers.csv")



with insight:
    customer_insight   = pd.read_csv("customers_insight.csv")
    customersInfo   = pd.read_csv("customers.csv")
    loans       = pd.read_csv("loans.csv")
    repaidLoans = loans[loans['loan_status'] == 5]
    defaultedLoans = loans[loans['loan_status'] == 6]
    st.text('Loans and Customers Statistics')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Customers", len(customer_insight))
    with col2:
        st.metric("Loans", len(loans))
    with col3:
        st.metric("Repaid Loans", len(repaidLoans))
    with col4:
        st.metric("Defaulted Loans", len(defaultedLoans))
    
    st.divider()
    col1, col2 = st.columns(2)
    nomalizeData2()
    with col1:
        st.caption("Customers Segmentatioin Based on their Similarities")
        customers = pd.read_csv("customers_insight.csv")
        first_nomolized = customers.groupby(['cluster']).size().reset_index(name='count')
        first_nomolized['priority'] = first_nomolized['cluster'].map({0:'Low Priority', 1:'Medium Priority', 2:'High Priority'})
        st.table(first_nomolized[['count','priority']])
        st.caption("Customer Distribution Based On  their Similarities")
        st.scatter_chart(customers,x='id' , y='cluster')
    
    with col2:
        st.caption("Customer Segmentation Based on their Score")
        customers = pd.read_csv("customers_insight.csv")
        customers['final_cluster']  = customers['final_priority'].map({'Low Priority':0,'Medium Priority':1,'High Priority':2})
        second_nomolized = customers.groupby(['final_cluster']).size().reset_index(name='count')
        second_nomolized['priority'] = second_nomolized['final_cluster'].map({0:'Low Priority', 1:'Medium Priority', 2:'High Priority'})
        st.table(second_nomolized[['priority','count']])
        st.caption("Score Distribution By Customer")
        st.scatter_chart(customers, x = 'id', y = 'score')
        
        
        





    #analyzeButton = st.button("Start analyzing")
    # if analyzeButton :
    #     progress_bar = st.progress(0)
    #     customers   = pd.read_csv("customers.csv")
    #     customers['dob']    = pd.to_datetime(customers['dob'])
    #     today               = datetime.today()
    #     customers['age']    = (today - customers['dob']).astype('<m8[Y]').astype(int)
    #     customers['year_range']     = customers['age'].apply(age_range)
    #     customers.to_csv("customers.csv", index=False)
        
    #     progress_bar.progress(100)

    

    #datasets()
    #customerAnalyzer()
    #nomalizeData2()




