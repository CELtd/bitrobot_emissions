#!/bin/bash

gcloud builds submit --tag gcr.io/cel-streamlit/bitrobot-emissions

gcloud run deploy bitrobot-emissions \
  --image gcr.io/cel-streamlit/bitrobot-emissions \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --min-instances=0 \
  --port=8501
