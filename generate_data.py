import pandas as pd
import numpy as np

# Create fake employee data
np.random.seed(42)

data = pd.DataFrame({
    "Age": np.random.randint(22, 60, 200),
    "JobSatisfaction": np.random.randint(1, 5, 200),
    "YearsAtCompany": np.random.randint(0, 20, 200),
    "WorkLifeBalance": np.random.randint(1, 5, 200),
    "OverTime": np.random.randint(0, 2, 200),
    "Salary": np.random.randint(30000, 100000, 200),
    "Department": np.random.randint(1, 5, 200),  # 1 = HR, 2 = Sales, etc.
    "Attrition": np.random.randint(0, 2, 200)  # 0 = No, 1 = Yes
})

# Save it as a CSV file
data.to_csv("employee_data.csv", index=False)

print("✅ Fake employee data created!")
