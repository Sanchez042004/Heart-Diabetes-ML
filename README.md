# Heart Disease & Diabetes with Machine Learning
[![Pub](https://img.shields.io/badge/-Jupyter-F37626?style=plastic&logo=jupyter&logoColor=white)](https://pub.dev/packages/dart_code_linter)
[![Pub](https://img.shields.io/badge/Python-3670A0?style=plastic&logo=python&logoColor=ffdd54)](https://pub.dev/packages/dart_code_linter)

Machine learning presents an extraordinary opportunity to elevate healthcare standards by enabling early disease detection. By developing precise models for identifying heart disease and diabetes, we can initiate timely interventions, personalized treatment plans, and proactively manage health concerns.

## Data Collection

To begin the process, I collected extensive patient data and outcomes from various medical records and health databases. These datasets, obtained from sources like Kaggle and Google, provided a diverse range of information crucial for training robust machine learning models.

<p align="center">
  <img src="images/Kaggle Heart.png" alt="Kaggle Heart Image" width="700" height="320">
</p>

<p align="center">
  <img src="images/Kaggle Diabetes.png" alt="Kaggle Diabetes Image" width="700" height="270">
</p>

## ETL Process: Extract, Transform, Load

The ETL (Extract, Transform, Load) process is a critical step in preparing data for machine learning and analysis. It begins with extracting raw data from various sources, followed by transforming this data through cleaning and conversion processes to ensure it is suitable for analysis. 

<p class="imagenes" align="center">
  <img src="images/ETL.jpg" alt="ETL Process Image" width="700" height="320">
</p>

### Extract
Initially, I extracted raw data from the collected sources, encompassing patient demographics, medical history, diagnostic tests, and disease outcomes. 

For example this is an extract code for heart disease:
```python
heart_disease_counts.plot(kind="bar", stacked=True)
plt.title("Heart Disease Cases by Age Group & Gender")
plt.xlabel("Age Group")
plt.ylabel("Number of Cases")
plt.xticks(rotation=45)
plt.legend(title="Gender")
plt.tight_layout()
plt.show()
```
### Transform
The extracted data was then subjected to rigorous transformation processes. This involved cleaning the data by handling missing values ​​and removing irrelevant information. Additionally, I converted categorical data to numerical format for effective analysis. 

Subsequently, I used the Matplotlib library to generate graphs and visualizations from the numerical data, allowing for better understanding and analysis of the information.


<p align="center">
  <img src="images/Correlation.png" alt="Correlation Image" width="400" height="320">
  <img src="images/Cases.png" alt="Cases Image" width="400" height="320">
</p>


### Load
Once transformed, the processed data was loaded into the machine learning pipeline, ready for further processing and model development. To perform the loading process, Docker can be used for containers, and the file can be hosted in a database within the container. Additionally, Power BI can be used to visualize the data.


## Data Preprocessing

Following the ETL process, the preprocessed data underwent normalization to ensure uniform data scalability. This step is crucial for preventing certain features from dominating the model training process due to differences in their scales.

## Data Splitting

To evaluate the performance of our models accurately, I split the dataset into training and testing sets. Typically, 70-80% of the data was allocated for training, while the remaining 20-30% was reserved for testing.

## Model Selection

I carefully selected appropriate algorithms, including decision trees and logistic regression, after thorough testing to determine the models that best fit our dataset. Various parameters were fine-tuned to enhance model performance.

```jupyter
x = pd.get_dummies(x, columns=categories.keys())
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
logistic_classifier = LogisticRegression()
logistic_classifier.fit(x_train, y_train)
```

## Disease Detection

Utilizing the trained models, I implemented disease detection algorithms capable of accurately identifying heart diseases or diabetes based on new patient data.

```markdown
Enter age: 30
Enter sex (M/F): M
Enter chest pain type (ATA/NAP/ASY/TA): NAP
Enter resting blood pressure: 180
Enter cholesterol: 130
Enter fasting blood sugar: 1
Enter resting ECG (Normal/ST/LVH): Normal
Enter max heart rate: 75
Enter exercise angina (N/Y): Y
Enter oldpeak: 1.3
Enter ST Slope (Up/Flat/Down): Down
¡Heart Disease Detected!
```

## Main Causes of Heart Disease & Diabetes

Understanding the primary risk factors for heart disease and diabetes is crucial for effective prevention and management:

- **High Blood Pressure:** Elevated blood pressure increases the strain on the heart and blood vessels, significantly raising the risk of coronary artery disease and heart attacks.
  
- **High Cholesterol:** Elevated levels of LDL cholesterol can lead to the buildup of plaque in the arteries, narrowing blood flow and increasing the risk of heart disease.
  
- **Obesity:** Obesity contributes to diabetes by promoting insulin resistance, where cells become less responsive to insulin signals, leading to elevated blood sugar levels. Over time, the pancreas may struggle to produce enough insulin to compensate, resulting in type 2 diabetes.

## Tools & Environment

I employed the following tools and environment for our machine learning project:

- **Programming Language:** Python
  
- **Libraries:** Pandas, Matplotlib, NumPy, and Scikit Learn

- **Integrated Development Environments (IDEs):**
  - Visual Studio Code
  - Jupyter Notebook
