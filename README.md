# A04-eto25001
A04: Model Interpretability (xAI)

Loan Status Model Interpretation

This project builds a classification model to predict loan approval status using the loan dataset. The main purpose of the code is to train a model, identify the top 5 most important features using permutation importance, and create interpretation plots such as feature importance boxplots and ICE/PDP plots.

Files

- `A04-model.py` — main Python script
- `train_loan_imbalanced.csv` — input dataset
- `figures/` — folder where output plots are saved

Requirements

Install the required Python packages:
pip install pandas matplotlib numpy scikit-learn pycebox

How to Run

1. Download or clone this GitHub repository to your computer.
2. Make sure the input file is included in the repository folder: train_loan_imbalanced.csv
3. Open `A04-model.py` in Visual Studio Code or another Python editor.
4. Update the file path in the code so it points to the dataset location on your computer.
For example, if the CSV file is in the same folder as the Python script, use: df = pd.read_csv("train_loan_imbalanced.csv")
If the file is in a different folder, update the path to match your computer.
5. Run the script from the terminal: python A04-model.py

Output

The script will:
1. Load and clean the loan dataset
2. Train a classification model using `Loan_Status` as the target variable
3. Use grid search to tune the model
4. Calculate permutation importance with 20 repeats
5. Save feature importance boxplots
6. Save ICE plots with PDP overlays for the top 5 features
All generated plots will be saved in the `figures/` folder.

Analysis:
- The x-axis in each partial dependence plot represents the values of that specific feature: credit history, urban property area, applicant income, co-applicant income, and loan amoun term. The Y-axis represents the model’s average predicted probability of loan approval and stays consistent across all plots.
- Credit history shows the strongest relationship with loan approval. Applicants with stronger (longer?) credit history are predicted to have a higher probability of being approved.
- Co-applicant income and loan amount term show smaller changes in predicted approval probability, which suggests that income and loan term matter but aren't as dominant as credit history in this model.
- Property area urban and applicant income had relatively flat partial dependence plots, which suggests that changing those feature values does not strongly affect the model’s prediction.
- One thing that surprised me is that the model does not appear to rely equally on all financial variables and instead it seems to place more importance on reliability (credit history and if you have a dependable co-applicant income) over just the main applicant's income.