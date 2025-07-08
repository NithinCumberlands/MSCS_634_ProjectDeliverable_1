# Data Mining Capstone – Deliverable 1

## Dataset Summary
I selected the **Titanic Passenger Survival** dataset from Kaggle, which contains **891** records and **12** core features: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, and Embarked. This real-world dataset includes missing values (Age, Cabin), duplicates, and skewed distributions—making it ideal for practicing data cleaning and exploratory analysis.

## Key Insights from EDA
- **Age Distribution:** I discovered that Age is right-skewed, with clusters of children and middle-aged adults. This informed my decision to impute missing values using median by Pclass and Sex.
- **Fare Distribution:** Fare was heavily skewed by a few very expensive tickets. Applying a log transform (`log1p`) normalized the distribution for future modeling steps.
- **Survival by Sex and Class:** Females had a significantly higher survival rate than males, and first-class passengers survived at higher rates than those in second and third classes. These patterns indicate that `Sex` and `Pclass` will be strong predictors.
- **Engineered Features:** I created `FamilySize` (SibSp + Parch + 1) and `IsAlone` flags, both of which showed promising correlations with survival.

## Major Steps in Data Cleaning & Exploration
1. **Missing Value Treatment**  
   - Imputed missing `Age` values within each (`Pclass`, `Sex`) group using the median.  
   - Dropped the `Cabin` column due to over 75% missing entries.  
   - Filled the two missing `Embarked` values with the mode.

2. **Duplicate Handling**  
   - Checked for and removed any duplicate `PassengerId` entries.

3. **Feature Engineering**  
   - Added `FamilySize` and binary `IsAlone` features to capture group dynamics.

4. **Exploratory Visualization**  
   - Plotted histograms for `Age` and log-transformed `Fare`.  
   - Created count plots and bar charts to examine survival rates by `Sex` and `Pclass`.  
   - Generated a correlation heatmap for numeric features to identify relationships.

## Challenges & Resolutions
- **High Missingness in Cabin:** With over three-quarters of `Cabin` values missing, I chose to drop the column entirely to avoid unreliable imputations.
- **Chained-Assignment Warnings:** I replaced any `inplace=True` calls on Series with explicit reassignment to ensure future compatibility with Pandas 3.0.
- **Skewed Distributions:** To address extreme skew in `Fare`, I applied a `log1p` transformation, which improved symmetry and will benefit downstream models.

---

*This README provides an overview of my data preparation and exploratory analysis as the foundation for predictive modeling in Deliverable 2.*  
