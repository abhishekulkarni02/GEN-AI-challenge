import streamlit as st
import pandas as pd
from fpdf import FPDF
import os
from google import genai
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import openai
from io import BytesIO

# Set your OpenAI API Key

def detect_anomalies(df):
    anomalies = "" 
    # #Using ML:
    # if 'Amount' not in df.columns:
    #     return "No 'Amount' column found in the dataset. Please check your file."
    
    # amounts = df[['Amount']].dropna()
    # model = IsolationForest(contamination=0.05, random_state=42)
    # model.fit(amounts)
    # df['Anomaly'] = model.predict(amounts)
    # anomalies = df[df['Anomaly'] == -1]
    # anomalies = "\nUsing ML anomalies are:\n"+anomalies.to_string()


    required_columns = ["As_Of_Date", "GL_Secondary_Account", "No_Of_Account", "AggregateBaseDifference", "AggregateAdjustedDifference", "Status"]
    
    if not all(col in df.columns for col in required_columns):
        return "Missing required columns. Please check your file."
    
    df["As_Of_Date"] = pd.to_datetime(df["As_Of_Date"])  # Convert date column to datetime
    df.sort_values(by=["GL_Secondary_Account", "As_Of_Date"], inplace=True)  # Sort for historical context
    
    tbd_records = df[df["Status"] == "TBD"].copy()
    if tbd_records.empty:
        return "No TBD records found to analyze."
    
    anomalies_text = ""
    anomaly_results = ""
    grouped = tbd_records.groupby("GL_Secondary_Account")
    

    for account, group in grouped:
        historical_data = df[(df["GL_Secondary_Account"] == account) & (df["Status"] != "TBD")]

        prompt = """You are an AI that detects financial anomalies in transactions.\n
Analyze the following financial transactions and determine if they are anomalies (they may or may not be anomalies).
Output 'Yes' for anomalies and 'No' for normal records.\n
Display the record only if 'Yes' else leave empty.
Output format:  As_Of_Date : <Date>
                GL_Secondary_Account: <account>(start 10 letters)
                AggregateBaseDifference: <value>
                AggregateAdjustedDifference: <value>
                Reason: <reason> (one line)
                Anomaly Status : <status>
            """

        if not historical_data.empty:
            prompt += "Historical Data for GL_Secondary_account ("+account+"):\n"
            for _, row in historical_data.tail(5).iterrows():  # Use last 5 historical records
                prompt += f"Date: {row['As_Of_Date'].date()}, Base Diff: {row['AggregateBaseDifference']}, Adjusted Diff: {row['AggregateAdjustedDifference']}, Status: {row['Status']}\n"
        else:
            prompt += "No historical data available for this account.\n"
        
        prompt += "\nNew Records to Analyze:\n"
        for _, row in group.iterrows():
            prompt += f"Date: {row['As_Of_Date'].date()}, No. of Accounts: {row['No_Of_Account']}, Base Diff: {row['AggregateBaseDifference']}, Adjusted Diff: {row['AggregateAdjustedDifference']}\n"
        
        client = genai.Client(api_key="your api key")

        print(prompt)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        anomaly_results += response.text + "\n"

    return anomaly_results

    # anomalies_text="None"
    # if 'Amount' not in df.columns:
    #     return "No 'Amount' column found in the dataset. Please check your file."
    
    # amounts = df[['Amount']].dropna().values.flatten().tolist()
    # prompt = f"Identify anomalies (if any) in these transaction amounts: {amounts}.\n The output should be in format: Index & Amount."


    # response = client.models.generate_content(
    #     model="gemini-2.0-flash",
    #     contents=prompt,
    # )
    
    # anomalies_text = response.text
    # return anomalies+"\n\n\n using GenAI anomalies are:\n "+anomalies_text


def generate_pdf(df, summary_text, anomalies_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary_text)
    pdf.ln(10)
    pdf.multi_cell(0, 10, "Possible anomalies:\n" + anomalies_text)
    
    # Generate and save graphs for each account
    for account in df["GL_Secondary_Account"].unique():
        account_data = df[df["GL_Secondary_Account"] == account]
        if account_data.empty:
            continue
        
        plt.figure(figsize=(7, 8))
        plt.plot(account_data["As_Of_Date"], account_data["AggregateAdjustedDifference"], marker='o', label='Adjusted Diff', color='red')
        plt.xlabel("Date")
        plt.ylabel("Difference")
        plt.title(f"Account: {account}")
        plt.legend()
        for i, txt in enumerate(account_data["AggregateAdjustedDifference"]):
            plt.annotate(f"{txt:.2f}", (account_data["As_Of_Date"].iloc[i], txt), textcoords="offset points", xytext=(0,-10), ha='center')
        
        img_path = f"{account}_plot.png"
        plt.savefig(img_path)
        plt.close()
        plt.ylim(min(account_data["AggregateBaseDifference"].min(), account_data["AggregateAdjustedDifference"].min()) - 10, 
                 max(account_data["AggregateBaseDifference"].max(), account_data["AggregateAdjustedDifference"].max()) + 10)  # Allow negative values
        
        pdf.add_page()
        pdf.image(img_path, x=10, y=30, w=180)
        os.remove(img_path)
    
    pdf_path = "Report.pdf"
    pdf.output(pdf_path)
    return pdf_path

    
st.title("Reconciliation - Anomaly Detection")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xls", "xlsx"])

if uploaded_file is not None:
    st.write("File uploaded successfully!")
    if st.button("Run Algorithm"):
        df = pd.read_excel(uploaded_file)
        summary_text = df.describe().to_string()
        anomalies_text = detect_anomalies(df)
        pdf_path = generate_pdf(df,summary_text, anomalies_text)
        
        st.subheader("Results")
        st.write("Possible anomalies:")
        st.text(anomalies_text)
        
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
            st.download_button(
                label="Download Results as PDF",
                data=pdf_bytes,
                file_name="Report.pdf",
                mime="application/pdf"
            )
        os.remove(pdf_path)