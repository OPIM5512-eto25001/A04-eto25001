import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from pycebox.ice import ice, ice_plot

df = pd.read_csv(r"C:/Users/Marisa/Downloads/train_loan_imbalanced.csv")
# create figures folder if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')

df.head()

print(df.shape)
print(df.info())
print(df['Loan_Status'].value_counts())
print(df.isnull().sum())

# replacing null values with median (numerical) or most common (categorical)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# identify categorical cols
categorical_cols = categorical_cols.drop(['Loan_Status', 'Loan_ID'])

# fill missing numeric values with median
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# fill missing categorical values with most common
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# checking work
print(df.isnull().sum())

# target variable
y = df['Loan_Status']

# feature variables
X = df.drop(columns=['Loan_Status', 'Loan_ID'])

# check dimensions
print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)

X.head()

# convert categorical -> indicator variables
X = pd.get_dummies(X, drop_first=True)

print(X.shape)
X.head()

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# define model & parameter grid
rf_model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# fit model
grid_search.fit(X_train, y_train)

# best model
best_model = grid_search.best_estimator_

print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Score:")
print(grid_search.best_score_)

# generate predictions
y_pred = best_model.predict(X_test)

# evaluate model performance
print("Test Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# find permutation importance
perm_importance = permutation_importance(
    best_model,
    X_test,
    y_test,
    n_repeats=20,
    random_state=42
)

# store results
importance_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance Mean': perm_importance.importances_mean
})

# sort by importance
importance_df = importance_df.sort_values(
    by='Importance Mean',
    ascending=False
)

# find top 5 features
top5_features = importance_df.head(5)['Feature'].values

print("Top 5 Features:")
print(top5_features)

top5_indices = [list(X_test.columns).index(f) for f in top5_features]

plt.figure(figsize=(10, 6))

plt.boxplot(
    perm_importance.importances[top5_indices].T,
    tick_labels=top5_features
)

plt.xticks(rotation=45)
plt.ylabel("Permutation Importance")
plt.title("Top 5 Feature Importances (20 Repeats)")
plt.tight_layout()
plt.savefig(f'figures/Top 5 Feature Importances Boxplots.png', dpi=300, bbox_inches='tight')
plt.show()


# adding in separate box plots since the first chart makes everything except credit history show up real bad
fig, axes = plt.subplots(1, 5, figsize=(20, 5))

for ax, feature, idx in zip(axes, top5_features, top5_indices):
    
    ax.boxplot(perm_importance.importances[idx])
    
    ax.set_title(feature)
    ax.set_ylabel("Importance")

plt.suptitle("Individual Feature Importance Boxplots", fontsize=14)
plt.tight_layout()
plt.savefig(f'figures/Individual Feature Importance Boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# Create ICE plots with PDP overlay for top 5 features
for feature in top5_features:

    # ICE needs a dataframe
    train_X_df = pd.DataFrame(X_train, columns=X.columns)

    # Define a prediction function that returns the probability of 'Y'
    def predict_proba_y(X_data_for_pred):
        # Ensure X_data_for_pred is a DataFrame with original column names
        if not isinstance(X_data_for_pred, pd.DataFrame):
            X_data_for_pred = pd.DataFrame(X_data_for_pred, columns=X.columns)
        # Get the index of the 'Y' class
        y_class_idx = list(best_model.classes_).index('Y')
        return best_model.predict_proba(X_data_for_pred)[:, y_class_idx]

    # Create ICE data
    ice_df = ice(
        data=train_X_df,
        column=feature,
        predict=predict_proba_y # Use the wrapper function
    )

    # Plot ICE curves with PDP line
    ice_plot(
        ice_df,
        c='dimgray',
        linewidth=0.3,
        plot_pdp=True,
        pdp_kwargs={'linewidth': 2}
    )

    plt.title(f'ICE Plot with PDP Overlay: {feature}')
    plt.xlabel(feature)
    plt.ylabel('Predicted Probability of Loan Approval') # Update ylabel for consistency
    plt.tight_layout()
    plt.savefig(f'figures/ICE Plot with PDP Overlay_{feature}.png', dpi=300, bbox_inches='tight')
    plt.show()