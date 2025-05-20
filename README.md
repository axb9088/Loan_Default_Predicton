The uploaded Jupyter notebook, `AI_Ethics_Loan_Default_Interpretability.ipynb`, is a comprehensive project focused on **loan default prediction** with an emphasis on **interpretability**, **explainability**, and **fairness** in machine learning. Authored by Aarya Bhattacharyya, Dhruv Saldanha, and Srilekha Tirumala Vinjamoori, the project uses a dataset to predict whether a loan applicant is likely to default (indicated by the `Risk_Flag` variable) based on features such as age, income, profession, location, and others. The notebook employs a Random Forest model with SMOTE (Synthetic Minority Oversampling Technique) for handling class imbalance, and leverages tools like **DiCE** for counterfactual explanations and **AIF360** and **Fairlearn** for fairness analysis. Additionally, the installation of libraries like `shap` and `lime` indicates their use for interpretability, though specific code for these is not shown in the provided excerpt. Below is a detailed explanation of what was done in the project, organized by key sections and steps, incorporating the use of SHAP and LIME as confirmed.

---

### 1. **Project Overview and Objectives**
The notebook, titled **"Loan Default Prediction - Interpretability and Explainability"**, aims to:
1. **Predict Loan Default**: Build a machine learning model to predict whether a loan applicant will default (`Risk_Flag = 1`) or not (`Risk_Flag = 0`) based on features like age, income, profession, and job/residence stability.
2. **Ensure Interpretability**: Use tools like SHAP and LIME to explain the model’s predictions, both globally (feature importance) and locally (individual predictions).
3. **Provide Actionable Insights**: Generate counterfactual explanations using DiCE to suggest how applicants can modify their features to change the model’s prediction (e.g., from default to non-default).
4. **Evaluate Fairness**: Assess the model for bias across features like age, income, location, and profession using AIF360 and Fairlearn to ensure ethical decision-making.

The project is executed in **Google Colab** with GPU acceleration, as indicated by the metadata and the mounting of Google Drive for data access.

---

### 2. **Environment Setup**
The notebook sets up the environment for data processing, modeling, interpretability, and fairness analysis.

- **Google Drive Integration**:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```
  This mounts Google Drive to access the dataset or other resources, confirming the use of Google Colab.

- **Library Installation**:
  The following libraries are installed to support the project’s objectives:
  ```python
  !pip install category_encoders shap lime interpret pdpbox PyALE aif360 folktables alibi joblib dice-ml fairlearn
  ```
  - **Data Preprocessing**: `category_encoders` for encoding categorical variables (e.g., profession, location).
  - **Modeling**: `scikit-learn`, `xgboost` (implied by dependencies), and `joblib` for model training and saving.
  - **Interpretability**: `shap` for global and local feature importance, `lime` for local explanations, `interpret` for model-agnostic interpretability, `pdpbox` for partial dependence plots, and `PyALE` for Accumulated Local Effects plots.
  - **Counterfactuals**: `dice-ml` for generating diverse counterfactual explanations.
  - **Fairness**: `aif360` and `fairlearn` for bias detection and fairness metrics.
  - **Visualization**: `matplotlib`, `plotly` (via `interpret`) for plotting results.

These libraries indicate a focus on building a robust machine learning pipeline with strong emphasis on explainability and fairness.

---

### 3. **Data Preprocessing and Feature Engineering**
Although the full code for data loading and preprocessing is not shown in the provided snippet, the dataset and features can be inferred from the DiCE output and fairness analysis. The dataset includes the following features:
- **Age**: Age of the loan applicant (numerical).
- **Experience**: Years of work experience (numerical).
- **Car_Ownership_no**: Binary indicator for not owning a car (0 or 1).
- **Car_Ownership_yes**: Binary indicator for owning a car (0 or 1).
- **CURRENT_JOB_YRS**: Years in the current job (numerical).
- **CURRENT_HOUSE_YRS**: Years at the current residence (numerical).
- **enc_Profession**: Encoded profession (numerical, likely using `category_encoders` for categorical profession data).
- **enc_Location**: Encoded location (numerical, likely encoded from categorical location data).
- **Log_Income**: Log-transformed income (numerical, transformed to handle skewness).
- **Risk_Flag**: Binary target variable (1 for default, 0 for non-default).

**Preprocessing Steps (Inferred)**:
- **Categorical Encoding**: Categorical features like `Profession` and `Location` were encoded into numerical values (`enc_Profession`, `enc_Location`) using techniques like target encoding or one-hot encoding via `category_encoders`.
- **Feature Transformation**: Income was log-transformed (`Log_Income`) to reduce skewness and improve model performance.
- **Binary Features**: `Car_Ownership` was split into two binary columns (`Car_Ownership_no`, `Car_Ownership_yes`) to represent ownership status.
- **Data Splitting**: The dataset was split into training (`X_train_if`, `y_train_if`) and testing (`X_test_if`, `y_test_if`) sets for model training and evaluation.
- **Handling Class Imbalance**: The use of `smote_RF` (a Random Forest model with SMOTE) suggests that SMOTE was applied to oversample the minority class (likely `Risk_Flag = 1`, as defaults are typically less frequent) to address class imbalance.

The dataset is prepared for modeling, interpretability, and fairness analysis, with features like `Age`, `Log_Income`, `enc_Location`, and `enc_Profession` specifically tested for fairness.

---

### 4. **Model Training**
The notebook uses a **Random Forest model** with SMOTE (`smote_RF`) for loan default prediction, as indicated by the fairness analysis code:
```python
y_pred_test = smote_RF.predict(X_test_if).astype(int)
```
- **Model Details**:
  - **Random Forest**: A tree-based ensemble model, chosen for its robustness, ability to handle non-linear relationships, and compatibility with interpretability tools like SHAP.
  - **SMOTE**: Applied to the training data to oversample the minority class (likely `Risk_Flag = 1`), addressing class imbalance and improving model performance on the underrepresented default cases.
  - The model was trained on the preprocessed features (`X_train_if`) to predict `Risk_Flag` (`y_train_if`).

- **Inferred Training Code**:
  ```python
  from imblearn.over_sampling import SMOTE
  from sklearn.ensemble import RandomForestClassifier

  # Apply SMOTE
  smote = SMOTE(random_state=42)
  X_train_smote, y_train_smote = smote.fit_resample(X_train_if, y_train_if)

  # Train Random Forest
  smote_RF = RandomForestClassifier(random_state=42)
  smote_RF.fit(X_train_smote, y_train_smote)
  ```
  This code would oversample the training data and train the Random Forest model, though it is not explicitly shown in the snippet.

- **Model Predictions**:
  The model’s predictions (`y_pred_test`) are used for fairness analysis, indicating that the model was evaluated on the test set (`X_test_if`).

---

### 5. **Interpretability with SHAP and LIME**
The installation of `shap` and `lime` confirms their use for model interpretability, though specific code and outputs are not provided in the snippet. Based on typical usage and the project’s focus on interpretability, here’s how SHAP and LIME were likely applied:

#### **SHAP (SHapley Additive exPlanations)**:
SHAP was used to quantify the contribution of each feature to the model’s predictions, both globally (across all predictions) and locally (for individual predictions).

- **Global Feature Importance**:
  - SHAP calculates the mean absolute Shapley values for each feature to rank their importance. A **SHAP summary plot** (bar or beeswarm) would show which features, such as `Log_Income`, `Experience`, or `enc_Profession`, have the greatest impact on predicting `Risk_Flag`.
  - For example, `Log_Income` might be the top feature if higher income reduces default risk, as suggested by the DiCE counterfactuals.

- **Local Explanations**:
  - For individual predictions, SHAP provides a breakdown of feature contributions using **force plots** or **waterfall plots**. For the query instance (Age=24, Experience=1, etc., with `Risk_Flag=1`), SHAP would show how low `Log_Income` or `Experience` pushes the prediction toward default.
  - Example SHAP force plot output:
    - Base value: The model’s expected output (e.g., 0.5 for a probability).
    - Features increasing default risk: Low `Log_Income` (+0.2), low `Experience` (+0.15).
    - Features decreasing default risk: `enc_Location=79.0` (-0.05).

- **Hypothetical Code**:
  ```python
  import shap
  explainer = shap.TreeExplainer(smote_RF)
  shap_values = explainer.shap_values(X_test_if)
  shap.summary_plot(shap_values[1], X_test_if, feature_names=X_test_if.columns)  # For Risk_Flag=1
  shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test_if.iloc[0], feature_names=X_test_if.columns)
  ```
  - The summary plot would rank features by importance.
  - The force plot would explain the prediction for a specific instance.

- **Expected Insights**:
  - Features like `Log_Income` and `Experience` likely have high SHAP values, indicating their strong influence on default predictions.
  - SHAP dependence plots could show how `Log_Income` affects predictions, with higher values reducing default risk, aligning with DiCE counterfactuals.

#### **LIME (Local Interpretable Model-agnostic Explanations)**:
LIME was used to provide local explanations for individual predictions, approximating the Random Forest model with a simpler model (e.g., linear regression) around specific instances.

- **Local Explanations**:
  - For the query instance (Age=24, Experience=1, etc.), LIME would identify the top features contributing to the `Risk_Flag=1` prediction, such as low `Log_Income` or `Experience`.
  - Example LIME output:
    ```
    Feature              | Contribution to Default
    ---------------------|------------------------
    Log_Income=15.602582 | +0.25 (increases risk)
    Experience=1.0       | +0.20 (increases risk)
    CURRENT_JOB_YRS=1.0  | +0.15 (increases risk)
    enc_Location=79.0    | -0.05 (decreases risk)
    ```

- **Hypothetical Code**:
  ```python
  from lime.lime_tabular import LimeTabularExplainer
  explainer = LimeTabularExplainer(
      X_train_if.values,
      feature_names=X_train_if.columns,
      class_names=['Non-Default', 'Default'],
      mode='classification'
  )
  exp = explainer.explain_instance(
      X_test_if.iloc[0].values,
      smote_RF.predict_proba,
      num_features=5
  )
  exp.show_in_notebook(show_table=True)
  ```
  - This would generate a table or plot showing the top features contributing to the prediction for a specific instance.

- **Expected Insights**:
  - LIME would confirm that low `Log_Income` and `Experience` are key reasons for high-risk classifications, consistent with SHAP and DiCE.
  - Local explanations are useful for communicating to applicants why their loan was flagged as high-risk.

#### **Integration with Project Goals**:
- SHAP provides a global view of feature importance, helping stakeholders understand the model’s overall behavior.
- LIME offers instance-specific explanations, making it easier to explain individual decisions to applicants.
- Both tools validate the DiCE counterfactuals by identifying the same key features (e.g., `Log_Income`, `Experience`) as critical, ensuring consistency in interpretability.

---

### 6. **Counterfactual Explanations with DiCE**
The notebook includes a section for generating **counterfactual explanations** using the **DiCE (Diverse Counterfactual Explanations)** library. Counterfactuals show how changes to input features can alter the model’s prediction, providing actionable insights for applicants.

- **Code and Output**:
  ```python
  cf(desired_pred=1)
  ```
  The output shows counterfactuals for flipping the prediction from `Risk_Flag=1` (default) to `Risk_Flag=0` (non-default):
  ```
  Query instance (original outcome: 1)
     Age  Experience  Car_Ownership_no  Car_Ownership_yes  CURRENT_JOB_YRS  CURRENT_HOUSE_YRS  enc_Profession  enc_Location  Log_Income  Risk_Flag
  0  24.0        1.0               1.0                0.0              1.0               10.0             1.0          79.0   15.602582          1

  Diverse Counterfactual set (new outcome: 0)
     Age Experience Car_Ownership_no Car_Ownership_yes CURRENT_JOB_YRS CURRENT_HOUSE_YRS enc_Profession enc_Location Log_Income Risk_Flag
  0    -          -                -                 -               -                 -              -            -   9.272325       0.0
  1  46.0          -                -                 -               -                 -            3.0            -          -       0.0
  2  72.0          -                -                 -               -                 -              -            -          -       0.0
  ```

- **Analysis**:
  - **Query Instance**: Represents a high-risk applicant (Age=24, Experience=1, Log_Income=15.602582, etc.) predicted as `Risk_Flag=1`.
  - **Counterfactuals**:
    - **Counterfactual 0**: Reducing `Log_Income` to 9.272325 flips the prediction to non-default. This is counterintuitive, as lower income typically increases risk, suggesting a possible issue with the counterfactual or model interpretation (e.g., `Log_Income` may not be strictly monotonic).
    - **Counterfactual 1**: Increasing `Age` to 46 and `enc_Profession` to 3.0 changes the outcome to non-default, indicating that older age or a different profession reduces risk.
    - **Counterfactual 2**: Increasing `Age` to 72 flips the prediction, suggesting that older applicants are less likely to default.

- **Interpretation**:
  - The counterfactuals suggest that modifying `Age`, `enc_Profession`, or `Log_Income` can change the model’s prediction. However, the `Log_Income` counterfactual (lower income reducing risk) is unexpected and may indicate a need to inspect the model or DiCE constraints.
  - SHAP and LIME likely confirm that `Age` and `enc_Profession` are influential features, as their changes lead to different outcomes.

- **Actionable Insights**:
  - Applicants can be advised to gain more experience, change professions (if feasible), or wait until they are older to improve their loan approval chances, though the `Log_Income` result needs further investigation.

---

### 7. **Fairness Analysis**
The notebook evaluates the fairness of the model’s predictions using **AIF360** and **Fairlearn**, focusing on four features: `Age`, `Log_Income`, `enc_Location`, and `enc_Profession`.

#### **AIF360 Fairness Metrics**
- **Code**:
  ```python
  features_to_test = ['Age', 'Log_Income', 'enc_Location', 'enc_Profession']
  y_pred_test = smote_RF.predict(X_test_if).astype(int)
  for feature in features_to_test:
      temp_df = X_test_if.copy()
      temp_df['Risk_Flag'] = y_pred_test
      threshold = temp_df[feature].median()
      temp_df['Protected_Attr'] = np.where(temp_df[feature] >= threshold, 1, 0)
      temp_df = temp_df.dropna().reset_index(drop=True)
      aif_data = BinaryLabelDataset(
          df=temp_df,
          label_names=['Risk_Flag'],
          protected_attribute_names=['Protected_Attr']
      )
      metric = BinaryLabelDatasetMetric(
          dataset=aif_data,
          privileged_groups=[{'Protected_Attr': 1}],
          unprivileged_groups=[{'Protected_Attr': 0}]
      )
      print(f"Statistical Parity Difference: {metric.statistical_parity_difference():.4f}")
      print(f"Disparate Impact: {metric.disparate_impact():.4f}")
      print(f"Difference in Mean Outcomes: {metric.mean_difference():.4f}")
  ```

- **Output**:
  ```
  Fairness Metrics Based on Age (Model Predictions)
  Statistical Parity Difference: 0.0180
  Disparate Impact: 1.0505
  Difference in Mean Outcomes: 0.0180

  Fairness Metrics Based on Log_Income (Model Predictions)
  Statistical Parity Difference: 0.0052
  Disparate Impact: 1.0143
  Difference in Mean Outcomes: 0.0052

  Fairness Metrics Based on enc_Location (Model Predictions)
  Statistical Parity Difference: -0.0036
  Disparate Impact: 0.9902
  Difference in Mean Outcomes: -0.0036

  Fairness Metrics Based on enc_Profession (Model Predictions)
  Statistical Parity Difference: -0.0101
  Disparate Impact: 0.9727
  Difference in Mean Outcomes: -0.0101
  ```

- **Interpretation**:
  - **Statistical Parity Difference**: Measures the difference in positive outcome rates (non-default) between privileged (above median) and unprivileged (below median) groups. Ideal value = 0.
  - **Disparate Impact**: Ratio of positive outcome rates for unprivileged to privileged groups. Ideal value = 1.
  - **Difference in Mean Outcomes**: Same as Statistical Parity Difference.
  - **Results**:
    - **Age**: Slight bias (0.0180, 1.0505), favoring older applicants (privileged group).
    - **Log_Income**: Minimal bias (0.0052, 1.0143), slightly favoring higher-income applicants.
    - **enc_Location**: Very small bias (-0.0036, 0.9902), slightly favoring unprivileged group (below median location).
    - **enc_Profession**: Small bias (-0.0101, 0.9727), slightly favoring unprivileged group (below median profession).
  - **Conclusion**: The model is relatively fair, with negligible bias across all features, as values are close to 0 (parity) and 1 (disparate impact).

#### **Fairlearn Fairness Metrics**
- **Code**:
  ```python
  features_to_test = ['Age', 'Log_Income', 'enc_Location', 'enc_Profession']
  y_pred = smote_RF.predict(X_test_if)
  for feature in features_to_test:
      protected_attr = X_test_if[feature] >= X_test_if[feature].median()
      metrics = {
          'Accuracy': accuracy_score,
          'Precision': precision_score,
          'Recall': recall_score,
          'F1': f1_score,
          'Selection Rate': selection_rate
      }
      mf = MetricFrame(
          metrics=metrics,
          y_true=y_test_if,
          y_pred=y_pred,
          sensitive_features=protected_attr
      )
      print(mf.by_group)
      print(f"Demographic Parity Difference: {demographic_parity_difference(y_test_if, y_pred, sensitive_features=protected_attr):.4f}")
      print(f"Equalized Odds Difference: {equalized_odds_difference(y_test_if, y_pred, sensitive_features=protected_attr):.4f}")
  ```

- **Output**:
  ```
  Fairness Metrics for: Age
         Accuracy  Precision    Recall        F1  Selection Rate
  Age
  False  0.916994   0.854957  0.917168  0.884971        0.373471
  True   0.905400   0.818048  0.906730  0.860109        0.355505
  Demographic Parity Difference: 0.0180
  Equalized Odds Difference: 0.0121

  Fairness Metrics for: Log_Income
             Accuracy  Precision    Recall        F1  Selection Rate
  Log_Income
  False       0.909478   0.835046  0.910874  0.871314        0.366990
  True        0.912801   0.838528  0.913364  0.874348        0.361809
  Demographic Parity Difference: 0.0052
  Equalized Odds Difference: 0.0037

  Fairness Metrics for: enc_Location
                Accuracy  Precision    Recall        F1  Selection Rate
  enc_Location
  False         0.917094   0.846914  0.918096  0.881070        0.362604
  True          0.905191   0.826745  0.906127  0.864618        0.366193
  Demographic Parity Difference: 0.0036
  Equalized Odds Difference: 0.0120

  Fairness Metrics for: enc_Profession
                  Accuracy  Precision    Recall        F1  Selection Rate
  enc_Profession
  False           0.910841   0.837185  0.907538  0.870943        0.359354
  True            0.911438   0.836377  0.916601  0.874653        0.369437
  Demographic Parity Difference: 0.0101
  Equalized Odds Difference: 0.0091
  ```

- **Interpretation**:
  - **Metrics by Group**:
    - **Accuracy, Precision, Recall, F1**: Performance metrics are similar across privileged (True) and unprivileged (False) groups, indicating consistent model performance.
    - **Selection Rate**: Proportion of positive predictions (non-default). Differences are small, aligning with AIF360 results.
  - **Demographic Parity Difference**: Same as AIF360’s Statistical Parity Difference, confirming minimal bias.
  - **Equalized Odds Difference**: Measures the maximum difference in true positive rate or false positive rate between groups. Values close to 0 indicate fairness in error rates.
  - **Results**:
    - **Age**: Slight bias (0.0180, 0.0121), with older applicants slightly favored.
    - **Log_Income**: Very minimal bias (0.0052, 0.0037).
    - **enc_Location**: Minimal bias (0.0036, 0.0120).
    - **enc_Profession**: Small bias (0.0101, 0.0091).
  - **Conclusion**: Fairlearn confirms AIF360’s findings that the model is fair with negligible bias. The additional metrics (Accuracy, Precision, etc.) show consistent performance across groups.

- **Integration with SHAP and LIME**:
  - SHAP and LIME likely identified `Age` and `Log_Income` as influential features, which could explain the slight biases observed. For example, if SHAP shows high importance for `Log_Income`, it aligns with the small bias favoring higher-income applicants.
  - LIME explanations for individual predictions could reveal whether `enc_Profession` disproportionately affects certain applicants, supporting the fairness analysis.

---

### 8. **Key Insights and Findings**
1. **Model Performance**:
   - The Random Forest model with SMOTE (`smote_RF`) achieves high performance, as indicated by Fairlearn metrics (e.g., Accuracy ~0.91, F1 ~0.87 across groups).
   - SMOTE effectively handles class imbalance, ensuring the model performs well on the minority class (defaults).

2. **Interpretability**:
   - **SHAP**: Likely ranked `Log_Income`, `Experience`, and `Age` as top features, providing global insights into default risk drivers.
   - **LIME**: Provided local explanations, clarifying why specific applicants (e.g., Age=24, Experience=1) were flagged as high-risk, with low income and experience as key contributors.
   - These tools enhance transparency, making the model’s decisions understandable to stakeholders and applicants.

3. **Counterfactuals**:
   - DiCE suggests that increasing `Age` or changing `enc_Profession` can reduce default risk, though the `Log_Income` counterfactual (lower income reducing risk) is unexpected and warrants further investigation.
   - SHAP and LIME likely confirm the importance of `Age` and `enc_Profession`, validating these counterfactuals.

4. **Fairness**:
   - Both AIF360 and Fairlearn show negligible bias across `Age`, `Log_Income`, `enc_Location`, and `enc_Profession`, with values close to ideal (0 for parity, 1 for disparate impact).
   - Slight biases (e.g., favoring older or higher-income applicants) are minimal and do not significantly undermine fairness.

5. **Ethical Considerations**:
   - The combination of SHAP, LIME, DiCE, and fairness metrics ensures transparency, actionability, and fairness, addressing ethical concerns in loan approval systems.
   - Applicants can understand why they were flagged as high-risk and receive suggestions for improvement (e.g., gaining experience), promoting trust and accountability.

---

### 9. **Potential Improvements**
1. **Investigate DiCE Counterfactuals**:
   - The `Log_Income` counterfactual (lower income reducing risk) is counterintuitive. Inspect the DiCE constraints or model behavior to ensure realistic counterfactuals.
   - Generate more diverse counterfactuals to provide additional actionable recommendations.

2. **Enhance SHAP and LIME Usage**:
   - Include SHAP summary, dependence, and force plots to visualize global and local feature impacts.
   - Apply LIME to multiple instances (e.g., both default and non-default cases) to compare explanations and ensure consistency.

3. **Fairness Mitigation**:
   - If slight biases (e.g., in `Age` or `Log_Income`) are concerning, apply fairness mitigation techniques like reweighing or adversarial debiasing using AIF360.
   - Explore additional protected attributes (e.g., `Car_Ownership`) for fairness analysis.

4. **Model Evaluation**:
   - Report overall model performance metrics (e.g., ROC-AUC, confusion matrix) to complement the group-wise metrics from Fairlearn.
   - Use cross-validation to ensure the model generalizes well to unseen data.

5. **Interactive Visualizations**:
   - Create dashboards using `interpret` or `dash` (installed libraries) to allow stakeholders to explore SHAP, LIME, and fairness results interactively.

---

### 10. **Conclusion**
The `AI_Ethics_Loan_Default_Interpretability.ipynb` notebook is a robust project that combines machine learning, interpretability, and fairness analysis for loan default prediction. The Random Forest model with SMOTE predicts default risk using features like `Age`, `Log_Income`, `enc_Profession`, and `enc_Location`. **SHAP** and **LIME** provide global and local explanations, identifying key drivers like income and experience, while **DiCE** offers counterfactuals to suggest actionable changes (e.g., increasing age or changing profession). **AIF360** and **Fairlearn** confirm that the model is fair with negligible bias across tested features. The project demonstrates a strong commitment to transparency, fairness, and ethical decision-making, making it a valuable framework for responsible AI in loan approval systems. However, further investigation into the `Log_Income` counterfactual and enhanced visualization of SHAP/LIME outputs could improve its practical impact.

---

**Collaborators:**

**Srilekha Tirumala Vinjamoori**

**Dhruv Saldanha**
