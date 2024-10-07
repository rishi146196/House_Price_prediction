# House Price Prediction with Machine Learning

This repository focuses on predicting house prices by leveraging several machine learning algorithms. The project walks through essential stages such as data cleaning, feature engineering, model training, and evaluation, providing an in-depth analysis of each method’s performance.

## Project Goals

The primary goal of this project is to build and compare the performance of various machine learning models in predicting house prices. It aims to identify the model that best generalizes the data by exploring different approaches and optimizing their performance.

## Key Highlights

- *Data Preprocessing*: Handling missing values, encoding categorical variables, and scaling features to prepare data for machine learning models.
- *Exploratory Data Analysis*: Visualizing data distributions, correlations, and feature importance to gain insight into the underlying trends.
- *Machine Learning Algorithms*:
  - Multiple Regression
  - Decision Trees
  - Random Forest
  - Support Vector Machines (SVM)
  - Gradient Boosting (GBM)
  - XGBoost
- *Performance Evaluation*: Assessment of model performance using metrics like RMSE, MAE, and R², along with cross-validation.
- *Model Optimization*: Employing grid search for hyperparameter tuning to maximize model accuracy and reduce overfitting.

## Dataset

The dataset contains detailed information about houses, including features like the number of bedrooms, square footage, and location. You can download it [here](insert_link_to_dataset).

## Setup Instructions

To run the project locally, follow these steps:

1. Clone the repository:
   bash
   git clone https://github.com/your-username/house-price-prediction.git
   

2. Move into the project directory:
   bash
   cd house-price-prediction
   

3. Install the dependencies:
   bash
   pip install -r requirements.txt
   

## Running the Notebook

1. Open the Jupyter Notebook:
   bash
   jupyter notebook House_Price.ipynb
   

2. The notebook contains code to:
   - Load the dataset and preprocess the data
   - Perform exploratory data analysis
   - Train and test multiple machine learning models
   - Evaluate model performance using various metrics
   - Fine-tune models through hyperparameter optimization

## Machine Learning Models in Focus

- *Linear Regression*: A foundational model to establish a benchmark for performance.
- *Decision Trees & Random Forest*: These tree-based models capture non-linear relationships in the data and work well with minimal preprocessing.
- *Support Vector Machines (SVM)*: A powerful algorithm that works well with high-dimensional data.
- *Gradient Boosting & XGBoost*: Boosting algorithms that are known for their accuracy and performance in prediction tasks.

## Results

Each model's predictions are evaluated using metrics such as RMSE (Root Mean Squared Error) and R². The best-performing model is selected based on its accuracy and ability to generalize to unseen data.

## Contributions

If you’d like to contribute to this project, feel free to open issues or submit pull requests. Contributions of all kinds are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

---

Let me know if this suits your needs or if you want further adjustments!

## GUI
![Example Image](https://github.com/rishi146196/House_Price_prediction/blob/main/House_price_gui2.png?raw=true)

![Example Image](https://github.com/rishi146196/House_Price_prediction/blob/main/House_price_gui1.png?raw=true)

# HOW TO RUN GUI
To run a Streamlit app, you need to follow a series of steps. Here’s a complete guide along with the commands you'll need:

### Step-by-Step Guide to Running a Streamlit App

#### 1. **Install Streamlit**

If you haven’t installed Streamlit yet, you can do so using pip. Open your terminal or command prompt and run:

```bash
pip install streamlit
```

#### 2. **Create a Streamlit App**

1. Create a new Python file for your Streamlit app. You can name it something like `app.py`. Use a code editor to write your app. For example, you can create it in your terminal:

   ```bash
   touch app.py
   ```

2. Open `app.py` in your code editor and add your Streamlit code. Here’s a simple example:

   ```python
   import streamlit as st

   st.title("My First Streamlit App")
   st.write("Hello, world!")
   ```

#### 3. **Run the Streamlit App**

Navigate to the directory where your `app.py` file is located in your terminal and run the following command:

```bash
streamlit run app.py
```

#### 4. **Access the App**

Once you run the command, Streamlit will start a local server and provide you with a URL (usually `http://localhost:8501`). Open this URL in your web browser to see your Streamlit app in action.

### Summary of Commands

1. **Install Streamlit:**
   ```bash
   pip install streamlit
   ```

2. **Create a Streamlit app file:**
   ```bash
   touch app.py
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

### Stopping the App

To stop the Streamlit server, go back to the terminal where it’s running and press `Ctrl + C`.

### Additional Tips

- If you make changes to your `app.py`, the Streamlit app will automatically refresh in the browser.
- You can also run other Python scripts that are designed for Streamlit in the same way by replacing `app.py` with your script’s filename.

# Pickel file 
you need to create pickle file first, detail code share in above pkl file refer that.

