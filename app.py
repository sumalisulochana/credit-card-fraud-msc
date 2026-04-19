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
threshold = joblib.load("threshold.pkl")
#threshold = joblib.load("threshold.pkl")

# -------------------------
# CREATE ENCODERS (IMPORTANT)
# -------------------------
#categorical_cols = [
    #'GENDER','CAR','REALITY',
    #'INCOME_TYPE','EDUCATION_TYPE',
    #'FAMILY_TYPE','HOUSE_TYPE'
#]
#categorical_cols = [
    #'gender','car','reality',
    #'income_type','education_type',
#]
encoders = joblib.load("encoders.pkl")
#encoders = {}

#for col in categorical_cols:
    #le = LabelEncoder()
    #le.fit(df[col].astype(str))   # same categories as training
    #encoders[col] = le
    
# -------------------------
# NUMERICAL COLUMNS (FOR SCALING)
# -------------------------
numerical_cols = [
    'no_of_child','family_size','age',
    'years_employed','income'
]

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
        ui.input_numeric("income", "Income", 100000, min=0, max=2000000),
        ui.input_numeric("age", "Age", 35),
        ui.input_numeric("years_employed", "Years Employed", 5),
        ui.input_select("INCOME_TYPE", "Income Type", df['INCOME_TYPE'].unique().tolist()),

        ui.h4("🏠 Family Information"),
        ui.input_numeric("family_size", "Family Size", 2),
        ui.input_numeric("no_of_child", "No of Children", 0),
        ui.input_select("FAMILY_TYPE", "Family Type", df['FAMILY_TYPE'].unique().tolist()),
        ui.input_select("HOUSE_TYPE", "House Type", df['HOUSE_TYPE'].unique().tolist()),

        ui.input_select("EDUCATION_TYPE", "Education Type", df['EDUCATION_TYPE'].unique().tolist()),
        ui.h4("📱 Contact Information"),

        ui.input_select("flag_mobil", "Has Mobile", [1, 0]),
        ui.input_select("work_phone", "Work Phone", [1, 0]),
        ui.input_select("phone", "Phone", [1, 0]),
        ui.input_select("e_mail", "Email", [1, 0]),

        ui.h4("📅 Account Info"),
        ui.input_numeric("begin_month", "Months Since Start", 50),

        ui.input_select("model_choice", "Select Model",
                        ["Random Forest", "Logistic Regression"]),

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
        #encoded = {}
        #for col in categorical_cols:
            #value = getattr(input, col)()
            #encoded[col] = encoders[col].transform([str(value)])[0]
        encoded = {}

        # SIMPLE SAFE MAPPING
        #encoded['gender'] = 1 if input.GENDER() == 'M' else 0
        #encoded['car'] = 1 if input.CAR() == 'Y' else 0
        #encoded['reality'] = 1 if input.REALITY() == 'Y' else 0

# For now (to make app work)
        #encoded['income_type'] = 0
        #encoded['education_type'] = 0
        #encoded['family_type'] = 0
        #encoded['house_type'] = 0    
        encoded['gender'] = encoders['gender'].transform([str(input.GENDER())])[0]
        encoded['car'] = encoders['car'].transform([str(input.CAR())])[0]
        encoded['reality'] = encoders['reality'].transform([str(input.REALITY())])[0]
        encoded['income_type'] = encoders['income_type'].transform([str(input.INCOME_TYPE())])[0]
        encoded['education_type'] = encoders['education_type'].transform([str(input.EDUCATION_TYPE())])[0]
        encoded['family_type'] = encoders['family_type'].transform([str(input.FAMILY_TYPE())])[0]
        encoded['house_type'] = encoders['house_type'].transform([str(input.HOUSE_TYPE())])[0]    
            
        income_value = min(input.income(), 2000000)
        # Create dataframe
        new_data = pd.DataFrame({
            'id': [0],
            'gender': [encoded['gender']],
            'car': [encoded['car']],
            'reality': [encoded['reality']],
            'no_of_child': [input.no_of_child()],
            'family_type': [encoded['family_type']],
            'house_type': [encoded['house_type']],
            'flag_mobil': [int(input.flag_mobil())],
            'work_phone': [int(input.work_phone())],
            'phone': [int(input.phone())],
            'e_mail': [int(input.e_mail())],
            'family_size': [input.family_size()],
            'begin_month': [input.begin_month()],
            'age': [input.age()],
            'years_employed': [input.years_employed()],
            #'income': [input.income()],
            'income': [income_value],
            'income_type': [encoded['income_type']],
            'education_type': [encoded['education_type']]
        })

        return new_data

    # -------- RESULT --------
    @output
    @render.ui
    def result_box():
        if input.predict_btn() > 0:
            

            new_data = build_input()
            
            new_data = new_data[model_columns]
            
            #print("APP columns:", new_data.columns.tolist())
            
            #return ui.div(
            #    ui.h4("DEBUG DATA"),
            #    ui.p(f"{new_data.iloc[0].to_dict()}")
            #)
            #print("Income type input:", input.INCOME_TYPE())
            #print("Encoded:", encoders['income_type'].transform([str(input.INCOME_TYPE())]))

            # Scaling
            new_scaled = scaler.transform(new_data)
            #new_data_scaled = new_data.copy()

            
            if input.model_choice() == "Random Forest":
                #return ui.div(
                #ui.p(f"Columns: {new_data.columns.tolist()}"),
                #)
            # RF should use ORIGINAL data (no scaling)
                prob = rf_model.predict_proba(new_data)[0][1]
                
            

            else:
            # LR needs scaled data
                new_scaled = scaler.transform(new_data)
                prob = lr_model.predict_proba(new_scaled)[0][1]
            # RF should use ORIGINAL data (no scaling)
            
            if input.model_choice() == "Random Forest":
                
                #if prob > 0.35:
                if prob > threshold:
                    return ui.div("🚨 HIGH RISK FRAUD",
                                style="background-color:red; color:white; padding:15px; border-radius:10px; font-size:18px;")
                elif prob > (threshold * 0.2):
                    return ui.div("⚠️ MEDIUM RISK",
                                style="background-color:orange; padding:15px; border-radius:10px; font-size:18px;")
                else:
                    return ui.div("✅ LOW RISK",
                                style="background-color:green; color:white; padding:15px; border-radius:10px; font-size:18px;")
            else:    
                
                if prob > 0.4:
                    return ui.div("🚨 HIGH RISK FRAUD",
                                style="background-color:red; color:white; padding:15px; border-radius:10px; font-size:18px;")
                elif prob > 0.2:
                    return ui.div("⚠️ MEDIUM RISK",
                                style="background-color:orange; padding:15px; border-radius:10px; font-size:18px;")
                else:
                    return ui.div("✅ LOW RISK",
                                style="background-color:green; color:white; padding:15px; border-radius:10px; font-size:18px;")

    # -------- PROBABILITY --------
    @output
    @render.text
    def prob():
        if input.predict_btn() > 0:

            new_data = build_input()
            
            #new_data = X_test.iloc[[0]]
            new_data = new_data[model_columns]
            #new_scaled = scaler.transform(new_data)
            
            new_data = new_data.astype(float)

            
            
            if input.model_choice() == "Logistic Regression":
            # LR needs scaled data
                new_scaled = scaler.transform(new_data)
                prob = lr_model.predict_proba(new_scaled)[0][1]

            else:
            # RF should use ORIGINAL data (no scaling)
                prob = rf_model.predict_proba(new_data)[0][1]
                
            return f"Fraud Probability: {prob:.6f} ({prob*100:.4f}%)"

# -------------------------
# RUN APP
# -------------------------
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()