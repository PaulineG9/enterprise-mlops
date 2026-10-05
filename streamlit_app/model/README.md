# Model folder

This folder is intentionally empty in the repo except for this file --
the actual trained model artifacts get downloaded from Azure ML and
placed here manually (or via CI, see the note at the bottom).

## Download the model from Azure ML

From your terminal (with `az login` active and the `ml` extension installed):

```bash
az ml model download \
  --name asset-failure-risk-classifier \
  --version <version-number> \
  --resource-group rg-enterprise-mlops \
  --workspace-name mlw-enterprise-mlops \
  --download-path ./downloaded_model
```

Find `<version-number>` in Azure ML Studio -> Models ->
`asset-failure-risk-classifier` -> the version list.

## Place the files here

The download produces a folder that itself contains a subfolder named
after the model (something like `downloaded_model/asset-failure-risk-classifier/`).
Copy the **contents** of that inner folder directly into this `model/`
folder, so you end up with:

```
streamlit_app/model/
├── MLmodel
├── conda.yaml
├── model.pkl (or python_model.pkl, depending on the MLflow version that logged it)
├── python_env.yaml
└── requirements.txt
```

`app.py` checks for `model/MLmodel` specifically to confirm the model is
in place before trying to load it.

## Why this isn't automated (for now)

The model is only a few MB, so committing it directly to the repo (rather
than fetching it at runtime) is simplest and keeps the Streamlit app's
startup fast and free of any Azure credentials or network calls. If you
retrain and want to update the demo, just re-run the download command
above and re-commit the refreshed files -- there's no live connection to
Azure ML from the running app at all.