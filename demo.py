import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("✅ Libraries working fine!")

# Simple test: create fake metal concentration data
data = {
    "Lead (Pb)": [0.02, 0.05, 0.12],
    "Arsenic (As)": [0.01, 0.03, 0.08],
    "Cadmium (Cd)": [0.005, 0.01, 0.02]
}

df = pd.DataFrame(data)
print(df)

df.plot(kind="bar")
plt.title("Test Heavy Metal Data")
plt.show()
