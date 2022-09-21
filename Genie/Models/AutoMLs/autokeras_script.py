import autokeras as ak
import pandas as pd

model = ak.AutoModel(
    inputs=[ak.TimeseriesInput()],
    outputs=[ak.RegressionHead()],
    project_name="auto_model",
    max_trials=100,
    directory=None,
    objective="val_loss",
    tuner="greedy",
    overwrite=False,
    seed=None,
    max_model_size=None,
    # **kwargs
)
print(model)
# Read Data
df = pd.read_csv("../../../Datas/XAUUSD_dollar_bars.csv")

y = df.pop("close")
x = df.drop(columns=["date_time"])


print(df.columns)
model.fit(
    x=x,
    y=y,
    batch_size=32,
    epochs=10,
    callbacks=None,
    validation_split=0.2,
    validation_data=None,
    verbose=1,
    # **kwargs
)

