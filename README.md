# Financial-Data-Analysis-and-Trading-Strategy-Project
## Computational Finance Project 2
This project is to predict stock price returns in financial markets based on the application of machine learning models, then test trading strategies based on the return prediction models using BackTrader. The focus will be geared towards the basic steps that are required for carrying out a data analysis project, implementing the machine learning models and using existing tools to backtest trading strategies, followed by analysis of the advantages/disadvantages of various trading models.



Installation Requirements
For executing the project's Jupyter Notebook, ensure that Jupyter Notebook is installed on your system along with the necessary libraries.

Necessary Python Packages
The project requires the following Python libraries for successful execution:

Pandas (1.5.0): Essential for data manipulation and analysis tasks.
NumPy: Supports complex numerical operations on arrays and matrices.
Pyfolio: Facilitates in-depth performance and risk analysis of financial portfolios.
yfinance: Aids in retrieving historical market data from Yahoo Finance.
datetime: Crucial for processing and manipulating date and time data.
Backtrader: Provides a comprehensive framework for trading strategy development and backtesting.
Scipy: Utilizes the mstats module for managing extreme data values.
Scikit-learn: A vital library for various machine learning operations, including model selection, preprocessing, and evaluation metrics. This encompasses tools like StandardScaler, MinMaxScaler, LogisticRegression, RandomForestClassifier, and several performance metrics.
Certain Pyfolio functionalities are incompatible with Pandas versions above 2.0.

class CustomCSVData(bt.feeds.GenericCSVData):This class, an extension of Backtrader's GenericCSVData, handles custom CSV data formats for strategy application.
def test_accuracy(X_test: pd.DataFrame, y_test: np.ndarray, model) -> None: A dedicated function for assessing the accuracy of a machine learning model with test data.
def tune_hyperparams(model, model_type: str, X_Train, y_train) -> None:This function is crucial for fine-tuning the hyperparameters of specific machine learning models. 

Key Insights and Conclusions
This project showcases the efficacy of custom data classes and advanced optimization techniques in the realm of machine learning. The implementation of a custom CSV data class provides a versatile approach to data handling. Meanwhile, the strategic hyperparameter tuning process markedly enhances model performance, leading to more accurate and reliable outcomes.
