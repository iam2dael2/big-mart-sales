import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import OneHotEncoder

df_train = pd.read_csv("Train.csv")
df_test = pd.read_csv("Test.csv")
df_train["Source"] = "train"
df_test["Source"] = "test"
df = pd.concat([df_train, df_test], axis=0)

## Imputation
df.loc[df["Item_Visibility"] == 0, "Item_Visibility"] = df_train["Item_Visibility"].median()
df["Item_Weight"].fillna(df_train["Item_Weight"].mean(), inplace=True)
df["Outlet_Size"].fillna(df_train["Outlet_Size"].mode().values[0], inplace=True)

## Categorical Processing
df["Item_Fat_Content"].replace({"LF": "Low Fat", "reg": "Regular", "low fat": "Low Fat"}, inplace=True)
df.loc[df["Item_Identifier"].apply(lambda x: x[:2]) == "NC", "Item_Fat_Content"] = "No Consumed"

preprocessed_data = df.copy()
preprocessed_data.drop(['Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Identifier'], axis=1, inplace=True)

# One Hot Encoding
onehot = OneHotEncoder(handle_unknown="ignore")
columns_to_encode = ["Item_Fat_Content", "Item_Type", "Outlet_Identifier"]

# Train Test Split
train_data = preprocessed_data[preprocessed_data["Source"] == "train"]
test_data = preprocessed_data[preprocessed_data["Source"] == "test"]

# Input and Targets
## Preprocessing: One Hot Encoding
transformed = onehot.fit_transform(train_data[columns_to_encode])
X_train = train_data.drop(["Item_Outlet_Sales", "Source"] + columns_to_encode, axis=1)
new_X_train = np.concatenate((X_train.values, transformed.toarray()), axis=1)
Y_train = train_data["Item_Outlet_Sales"]

# Choose page
st.sidebar.write("@Copyright 2023")
st.sidebar.write("Created by Darell Hermawan")
page = st.sidebar.selectbox("Page", ("About Us", "Exploration Data Analysis", "Model Prediction"))

def about():
    # Introduction
    intro = """
    The BigMart Sales Data is a comprehensive dataset that provides information on sales transactions 
    from various BigMart stores. It is an ideal dataset for exploratory data analysis, regression 
    modeling, and predictive analytics. The dataset contains a wide range of variables related to 
    product attributes, store details, and sales performance.
    """
    st.header(":red[Introduction]")
    st.write(f"<div style='text-align: justify;'>{intro}</div>", unsafe_allow_html=True)

    # Source
    source = """
    The BigMart Sales Data was obtained from BigMart, a leading retail chain with multiple stores 
    across different locations. The dataset represents sales data collected over a specific period and 
    captures information about various products sold in the stores.
    """
    st.header(":red[Source]")
    st.write(f"<div style='text-align: justify;'>{source}</div>", unsafe_allow_html=True)

    # Dataset
    st.header(":red[Dataset]")
    st.dataframe(df.head())
    st.markdown(f"Shape of the dataset: **{df.shape[0]}** rows and **{df.shape[1]}** columns.")

    # Content
    content = """
    The dataset comprises both numerical and categorical variables, offering a rich set of features for analysis. The key attributes in the dataset include:\n
    1. Item Identifier: A unique identifier for each product.
    2. Item Weight: The weight of the product.
    3. Item Fat Content: The fat content level in the product (low, regular, high).
    4. Item Visibility: The percentage of total display area allocated to the product in the store.
    5. Item Type: The category to which the product belongs (e.g., fruits, vegetables, household).
    6. Item MRP (Maximum Retail Price): The maximum price at which the product is sold.
    7. Outlet Identifier: A unique identifier for each store.
    8. Outlet Establishment Year: The year in which the store was established.
    9. Outlet Size: The size of the store (small, medium, large).
    10. Outlet Location Type: The type of location where the store is situated (urban, rural).
    11. Outlet Type: The type of outlet format (grocery store, supermarket, etc.).
    12. Item Outlet Sales: The sales of the product in the respective store (target variable).
    13. Source: Source of data (train/test).
    """
    st.header(":red[Content]")
    st.write(f"<div style='text-align: justify;'>{content}", unsafe_allow_html=True)

    # Purpose
    purpose = """
    The BigMart Sales Data was collected with the aim of analyzing sales patterns, identifying key 
    factors influencing sales, and developing predictive models to forecast future sales. It is 
    particularly useful for understanding the impact of various product and store attributes on sales 
    performance.
    """
    st.header(":red[Purpose]")
    st.write(f"<div style='text-align: justify;'>{purpose}", unsafe_allow_html=True)

def explore():
    # Shape
    st.subheader(":red[Shape]")
    st.markdown(f"**{df.shape[0]}** rows x **{df.shape[1]}** columns")

    col_1, col_2 = st.columns(2)
    with col_1:
        # Data type
        st.subheader(":red[Checking Data Types]")
        dtypes_df = pd.DataFrame(df.dtypes, columns=["Type"])
        st.write(dtypes_df)
    with col_2:
        # missing values
        st.subheader(":red[Checking Missing Types]")
        na_df = pd.DataFrame(df.isnull().sum(), columns=["Count"])
        st.write(na_df)
    
    na_col = st.selectbox("Choose which column is missing on the data to show",
                            ("Item_Weight", "Outlet_Size", "Item_Outlet_Sales")
                            )
    df_isna = df[df[na_col].isna()]
    st.dataframe(df_isna.head())
    if na_col == "Item_Outlet_Sales":
        st.error("All data whose source is 'test' has missing values on 'Item_Outlet_Sales'")
    elif na_col == "Outlet_Size":
        st.success(
            """
            Only these three outlets have unknown value on "Outlet_Size":\n
            1. OUT010
            2. OUT045
            3. OUT017
            """
        )
    else:
        st.warning("Nothing special.")    

    feature_type = st.selectbox("Which preferred type of features to explore?", ("None", "Numerical", "Categorical"))
    if feature_type == "Numerical":
        st.subheader(":red[Statistics Summary]")
        st.dataframe(df.describe())
        # Distribution Plot
        st.subheader(":red[Distribution Plot of Features]")
        kind = st.selectbox("Kind", ("Histogram", "Box Plot"))
        if kind == "Histogram":
            # Distribution plot: histogram
            dis_plot = df_train.hist(figsize=(12, 8), layout=(2, 3), color="tomato");
            st.pyplot(dis_plot[0][0].figure)
        else:
            # Distribution plot: boxplot
            num_cols = list(df.dtypes[(df.dtypes) != "O"].index)
            fig1, axs1 = plt.subplots(2, 3, figsize=(12, 8))

            count = 0
            for i in range(2):
                for j in range(3):
                    if count < len(num_cols):
                        axs1[i, j].boxplot(df[num_cols[count]])
                        axs1[i, j].set_title(num_cols[count])
                        count += 1
            st.pyplot(fig1)

        # Correlation map
        num_cols = list(df.dtypes[(df.dtypes) != "O"].index)
        st.subheader(":red[Correlation Map]")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.heatmap(df[num_cols].corr(), annot=True, ax=ax2)
        # Display the plot using st.pyplot()
        st.pyplot(fig2)
        st.success("By numerical features, Item_MRP has the highest correlation on Item_Outlet_Sales")
    elif feature_type == "Categorical":
        # Statistic Summary
        st.subheader(":red[Statistics Summary]")
        st.dataframe(df.describe(include="O"))

        st.subheader(":red[Distribution Plot of Features]")
        cat_feature_select = st.selectbox("Feature",
                                            ("Item_Fat_Content", "Item_Type", "Outlet_Information")
                                            )
        if cat_feature_select == "Item_Fat_Content":
            # Item_Fat_Content
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            sns.countplot(x=df["Item_Fat_Content"], order=df["Item_Fat_Content"].value_counts().index, 
                        palette="flare", ax=ax3)
            ax3.set_title("Item_Fat_Content")
            st.pyplot(fig3)
            st.info("Most of sold sales come from foods, especially low fat one.")
        elif cat_feature_select == "Item_Type":
            # Item_Type
            fig4, ax4 = plt.subplots(figsize=(8, 4))
            sns.countplot(y=df["Item_Type"], order=df["Item_Type"].value_counts().index, 
                        palette="flare", ax=ax4)
            ax4.set_title("Item_Type")
            st.pyplot(fig4)
            st.info("Fruits and vegetables have the most sales in BigMart.")
        elif cat_feature_select == "Outlet_Information":
            # Outlet
            outlet_cols = df.describe(include="O").columns[-5:-1]
            fig5, axs5 = plt.subplots(2, 2, figsize=(10, 6))

            plt.suptitle("Outlet Information", fontsize=14)
            count = 0
            for i in range(2):
                for j in range(2):
                    sns.countplot(y=df[outlet_cols[count]], order=df[outlet_cols[count]].value_counts().index, ax=axs5[i, j])
                    axs5[i, j].set_title(outlet_cols[count])
                    count += 1
            plt.tight_layout()
            st.pyplot(fig5)
            st.info("Most of outlets' sales are uniformly distributed. More specificly, most of transactions come to outlet whose size is medium, location is third-tier, and type is supermarket type 1.")

def building():
    st.write(f"<div style='text-align: right;'>Choose which model and parameter you're looking in sidebar.</div>", unsafe_allow_html=True)
    algo = st.sidebar.selectbox("Algorithm", 
                                ("Linear Regression", "Support Vector Machine", "Decision Tree", "Random Forest", "XGBoost")
                               )
    
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score, KFold

    # Swith case algorithm
    st.header(":red[Model Building]")
    model = None
    fold_5 = KFold(n_splits=5, shuffle=True, random_state=42)
    if algo == "Linear Regression":
        model = LinearRegression()
    elif algo == "Support Vector Machine":
        svr_kernel = st.sidebar.selectbox("Kernel", ("rbf", "linear", "sigmoid"))
        svr_C = st.sidebar.slider("C", 0, 10, 1)
        model = SVR(kernel=svr_kernel, C=svr_C)
    elif algo == "Decision Tree":
        dt_criterion = st.sidebar.selectbox("Criterion", ("squared_error", "friedman_mse", "absolute_error"))
        dt_max_depth = st.sidebar.selectbox("Max Depth", (None, 7, 9))
        model = DecisionTreeRegressor(criterion=dt_criterion, max_depth=dt_max_depth)
    elif algo == "Random Forest":
        rf_n_estimators = st.sidebar.slider("Estimators", 100, 2000, 100, 100)
        rf_max_depth = st.sidebar.selectbox("Max Depth", (None, 7, 9))
        model = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth)
    elif algo == "XGBoost":
        xgb_n_estimators = st.sidebar.slider("Estimators", 100, 2000, 100, 100)
        xgb_eta = st.sidebar.slider("Eta", 0.0, 1.0, 0.3, 1e-3)
        xgb_gamma = st.sidebar.slider("Eta", 0, 150, 0, 10)
        xgb_max_depth = st.sidebar.selectbox("Max Depth", (None, 7, 9))
        model = XGBRegressor(n_estimators=xgb_n_estimators,
                             eta=xgb_eta,
                             gamma=xgb_gamma,
                             max_depth=xgb_max_depth
                            )
    score = -cross_val_score(model, new_X_train, Y_train, scoring="neg_mean_squared_error", cv=fold_5).mean()
    st.subheader(":red[Result]")
    st.info(f"Mean squared error: {score}")

    model.fit(new_X_train, Y_train)
    st.subheader(":red[Result (Visualization)]")
    y_pred = model.predict(new_X_train)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Variation of Prediction**")
        fig6, ax6 = plt.subplots(figsize=(10, 8))
        ax6.scatter(y_pred, Y_train, color="pink", alpha=0.7)
        ax6.plot(Y_train, Y_train, color="red")
        ax6.set_xlim(0, 5000)
        st.pyplot(fig6)
    with col2:
        st.markdown("**Residual Plot**")
        fig7, ax7 = plt.subplots(figsize=(10, 8))
        sns.histplot(y_pred - Y_train, kde=True, ax=ax7)
        st.pyplot(fig7)
    return model

def predict(model):
    st.header(":red[Model Prediction]")
    # Input Processing
    item_weight = st.number_input("Insert an item's weight")
    item_fat_content = st.selectbox("Insert an item's fat content", ("Low Fat", "Regular", "No Consumed"))
    item_visibility = st.number_input("Insert an item's visibility")
    item_type = st.selectbox("Insert an item's type", 
                             ("Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods",
                              "Fruits and Vegetables", "Hard Drinks", "Health and Hygiene", "Household",
                              "Meat", "Others", "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods"
                             )
                            )
    item_mrp = st.number_input("Insert an item's MRP")
    outlet_identifier = st.selectbox("Insert an outlet's identifier", 
                             ("OUT010", "OUT013", "OUT017", "OUT018", "OUT019", "OUT027", "OUT035",
                              "OUT045", "OUT046", "OUT049")
                                    )
    inputs = pd.DataFrame([[item_weight, item_fat_content, item_visibility, item_type, item_mrp, outlet_identifier]],
                          columns=["Item_Weight", "Item_Fat_Content", "Item_Visibility", "Item_Type", "Item_MRP", "Outlet_Identifier"]
                         )
    st.dataframe(inputs)
    
    cat_transformed = onehot.transform(inputs[columns_to_encode])
    new_inputs = inputs.drop(columns_to_encode, axis=1)
    inputs = np.concatenate((new_inputs.values, cat_transformed.toarray()), axis=1)
    
    # Prediction
    preds = model.predict(inputs)[0]
    st.info(f"Outlet Sales of the Item: {preds}")

st.write(f"<h1 style='text-align: right;'>{page}</h1>", unsafe_allow_html=True)
if page == "About Us":
    about()
elif page == "Exploration Data Analysis":
    explore()
elif page == "Model Prediction":
    model = building()
    if st.sidebar.checkbox("Predict?"):
        predict(model)
