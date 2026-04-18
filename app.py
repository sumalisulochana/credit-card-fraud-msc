from shiny import App, ui, render
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

model_columns = [
'id','gender','car','reality','no_of_child','family_type',
'house_type','flag_mobil','work_phone','phone','e_mail',
'family_size','begin_month','age','years_employed',
'income','income_type','education_type'
]



# -------------------------
# LOAD DATA (FOR UI ONLY)
# -------------------------
df = pd.read_csv("merged_dataset.csv")

# -------------------------
# LOAD MODEL
# -------------------------
lr_model = joblib.load("lr_model.pkl")
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
#threshold = joblib.load("threshold.pkl")

# -------------------------
# CREATE ENCODERS (IMPORTANT)
# -------------------------
categorical_cols = [
    'GENDER','CAR','REALITY',
    'INCOME_TYPE','EDUCATION_TYPE',
    'FAMILY_TYPE','HOUSE_TYPE'
]

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    le.fit(df[col].astype(str))   # same categories as training
    encoders[col] = le

# UI
# -------------------------
app_ui = ui.page_sidebar(
#app_ui = ui.page_fluid(

    # -------- SIDEBAR --------
    ui.sidebar(

        ui.h4("👤 Personal Information"),
        ui.input_select("GENDER", "Gender", df['GENDER'].unique().tolist()),
        ui.input_select("CAR", "Own Car", df['CAR'].unique().tolist()),
        ui.input_select("REALITY", "Own House", df['REALITY'].unique().tolist()),

        ui.h4("💼 Financial Information"),
        ui.input_numeric("income", "Income", 100000),
        ui.input_numeric("age", "Age", 35),
        ui.input_numeric("years_employed", "Years Employed", 5),
        ui.input_select("INCOME_TYPE", "Income Type", df['INCOME_TYPE'].unique().tolist()),

        ui.h4("🏠 Family Information"),
        ui.input_numeric("family_size", "Family Size", 2),
        ui.input_numeric("no_of_child", "No of Children", 0),
        ui.input_select("FAMILY_TYPE", "Family Type", df['FAMILY_TYPE'].unique().tolist()),
        ui.input_select("HOUSE_TYPE", "House Type", df['HOUSE_TYPE'].unique().tolist()),

        ui.input_select("EDUCATION_TYPE", "Education Type", df['EDUCATION_TYPE'].unique().tolist()),

        ui.input_select("model_choice", "Select Model",
                        ["Logistic Regression", "Random Forest"]),

        ui.input_action_button("predict_btn", "🔍 Predict")
    ),

    # -------- MAIN CONTENT --------
    ui.div(

        ui.div(
            ui.h2("💳 Credit Card Fraud Detection System"),
            ui.p("Identify potential fraudulent customers using Machine Learning models"),
            style="text-align:center; margin-bottom:20px;"
        ),

        ui.navset_tab(

            ui.nav_panel("📊 Data Overview",
                ui.output_ui("overview")
            ),

            ui.nav_panel("📈 Data Analysis",
                ui.h3("📊 Fraud Analysis Dashboard"),
                ui.div(
                    ui.div(ui.output_plot("pie"), style="flex:1;"),
                    ui.div(ui.output_plot("hist"), style="flex:1;"),
                    style="display:flex; gap:20px;"
                ),
                ui.br(),
                ui.div(
                    ui.output_plot("box"),
                    ui.output_plot("fraud_income"),
                    style="width:50%; margin:auto;"
                )
            ),

            ui.nav_panel("🤖 Prediction",
                ui.h4("Prediction Result"),
                ui.output_ui("result_box"),
                ui.output_text("prob")
            )
        )
    )
)

# -------------------------
# SERVER
# -------------------------
def server(input, output, session):

    # -------- OVERVIEW --------
    @output
    @render.ui
    def overview():

        total_rows = df.shape[0]
        total_cols = df.shape[1]
        fraud_count = df['TARGET'].sum()
        non_fraud = total_rows - fraud_count

        return ui.div(

        # -------- KPI CARDS --------
            ui.div(
                ui.div(
                    ui.h4("📊 Total Records"),
                    ui.h2(f"{total_rows}"),
                    class_="card"
                ),
                ui.div(
                    ui.h4("📁 Total Features"),
                    ui.h2(f"{total_cols}"),
                    class_="card"
                ),
                ui.div(
                    ui.h4("🚨 Fraud Cases"),
                    ui.h2(f"{fraud_count}"),
                    class_="card red"
                ),
                ui.div(
                    ui.h4("✅ Non-Fraud"),
                    ui.h2(f"{non_fraud}"),
                    class_="card green"
                ),

                style="display:flex; gap:15px; margin-bottom:20px;"
            ),

            # -------- TABLE PREVIEW --------
            ui.h4("📄 Dataset Preview"),
            ui.HTML(df.head().to_html(index=False, classes="table table-striped"))
        )

    # -------- EDA --------
    @output
    @render.plot
    def pie():
        df['TARGET'].value_counts().plot.pie(
            autopct='%1.1f%%',
            labels=["Non-Fraud", "Fraud"])
        plt.title("Fraud vs Non-Fraud Distribution")
        return plt.gcf()

    @output
    @render.plot
    def hist():
        df['INCOME'].hist()
        plt.title("Customer Income Distribution")
        plt.xlabel("Income")
        plt.ylabel("Frequency")
        return plt.gcf()

    @output
    @render.plot
    def box():
        df.boxplot(column='INCOME')
        plt.title("Income Spread & Outliers")
        return plt.gcf()
    
    @output
    @render.plot
    def fraud_income():
        df.boxplot(column='INCOME', by='TARGET')
        plt.title("Income vs Fraud")
        plt.suptitle("")

    # -------- BUILD INPUT --------
    def build_input():

        # Encode categorical values
        encoded = {}
        for col in categorical_cols:
            value = getattr(input, col)()
            encoded[col] = encoders[col].transform([str(value)])[0]

        # Create dataframe
        new_data = pd.DataFrame({
            'id': [0],
            'gender': [encoded['GENDER']],
            'car': [encoded['CAR']],
            'reality': [encoded['REALITY']],
            'no_of_child': [input.no_of_child()],
            'family_type': [encoded['FAMILY_TYPE']],
            'house_type': [encoded['HOUSE_TYPE']],
            'flag_mobil': [0],
            'work_phone': [0],
            'phone': [0],
            'e_mail': [0],
            'family_size': [input.family_size()],
            'begin_month': [0],
            'age': [input.age()],
            'years_employed': [input.years_employed()],
            'income': [input.income()],
            'income_type': [encoded['INCOME_TYPE']],
            'education_type': [encoded['EDUCATION_TYPE']]
        })

        return new_data

    # -------- RESULT --------
    @output
    @render.ui
    def result_box():
        if input.predict_btn() > 0:
        

            new_data = build_input()
            new_data = new_data[model_columns]

            # Scaling
            new_scaled = scaler.transform(new_data)

            # 🔥 Select model
            if input.model_choice() == "Logistic Regression":
                model_used = lr_model
            else:
                model_used = rf_model

            prob = model_used.predict_proba(new_scaled)[0][1]

        # NEW THRESHOLD LOGIC
            if input.model_choice() == "Logistic Regression":
                if prob > 0.01:
                    return ui.div("🚨 HIGH RISK FRAUD",
                                style="background-color:red; color:white; padding:15px; border-radius:10px; font-size:18px;")
                elif prob > 0.005:
                    return ui.div("⚠️ MEDIUM RISK",
                                style="background-color:orange; padding:15px; border-radius:10px; font-size:18px;")
                else:
                    return ui.div("✅ LOW RISK",
                                style="background-color:green; color:white; padding:15px; border-radius:10px; font-size:18px;")
            else:
                if prob > 0.3:
                    return ui.div("🚨 HIGH RISK FRAUD",
                                style="background-color:red; color:white; padding:15px; border-radius:10px; font-size:18px;")
                else:
                    return ui.div("✅ LOW RISK",
                                style="background-color:green; color:white; padding:15px; border-radius:10px; font-size:18px;")
    # -------- PROBABILITY --------
    # -------- PROBABILITY --------
    @output
    @render.text
    def prob():
        if input.predict_btn() > 0:

            new_data = build_input()
            new_data = new_data[model_columns]
            new_scaled = scaler.transform(new_data)

            if input.model_choice() == "Logistic Regression":
                model_used = lr_model
            else:
                model_used = rf_model

            prob = model_used.predict_proba(new_scaled)[0][1]

            return f"Fraud Probability: {prob:.6f} ({prob*100:.4f}%)"

# -------------------------
# RUN APP
# -------------------------
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()