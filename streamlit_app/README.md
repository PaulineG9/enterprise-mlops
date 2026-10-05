# Asset Failure-Risk Predictor -- Streamlit Demo

A zero-ongoing-cost demo of the model trained in this project's
Databricks + Azure ML pipeline. Loads the model directly (no live Azure
endpoint), so it's free to host indefinitely on Streamlit Community
Cloud.

## Run locally

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```


## Deploy for free on Streamlit Community Cloud

1. Push this repo (including the populated `model/` folder) to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, and click "New app".
3. Point it at your repo, with **Main file path** set to
   `streamlit_app/app.py`.
4. Deploy. No secrets or Azure credentials are needed -- the app never
   talks to Azure at runtime, it only reads the bundled model files.

## Why this architecture

A live Azure ML Managed Online Endpoint (`src/deploy.py` in this repo)
bills continuously for the underlying compute, whether or not anyone
visits the demo -- typically tens to over a hundred dollars a month
depending on VM size. Loading the model directly inside this Streamlit
app avoids that entirely: the trained model is just a small file bundled
into the app, and Streamlit Community Cloud hosting is free. The
tradeoff is that this app can't demonstrate a live, callable production
API the way `deploy.py` + `scripts/test_endpoint.py` do -- if you want to
show that specific capability, deploy the Managed Online Endpoint
temporarily for a demo/screenshot, then delete it (`az ml online-endpoint
delete --name asset-risk-endpoint -y`) to stop the billing.