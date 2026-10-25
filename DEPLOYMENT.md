# Deploy ReviewMind to Streamlit Community Cloud

ReviewMind is prepared for Streamlit Community Cloud. The cloud service deploys from a GitHub repository; this project folder currently has no GitHub remote, so an app cannot be published until a repository is selected or created.

## Publish steps

1. Put the contents of this `ReviewMind` folder in a GitHub repository. Keep `app.py`, `requirements.txt`, `.streamlit/config.toml`, `src/`, and `data/` at the repository root.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) with a GitHub account that has admin access to the repository.
3. Choose **Create app**, select the repository and branch, and set the app file path to `app.py`.
4. In Advanced settings, choose a supported Python version compatible with the local environment (Python 3.11 or 3.12 is suitable).
5. Deploy. Community Cloud installs the packages in `requirements.txt` and reads `.streamlit/config.toml` for the app theme.

## Hosted behavior and privacy

- Uploaded files are processed by the hosted app while the session is active. Do not upload sensitive or personally identifying review data to a public demo.
- The sample dataset is synthetic. Uploaded content is not included in the repository.
- Community Cloud does not provide durable local disk for this prototype. SQLite analysis history may be reset when the app restarts or is rebuilt; durable multi-user history would need a persistent database service.
- Ollama is an optional local integration and is not available on Community Cloud by default. The hosted app continues to analyze data locally on its host and uses the labeled rule-based report fallback unless a reachable Ollama service is deliberately configured.
- No API keys or secrets are needed for the standard demo. Never commit `.env` or credentials.

See Streamlit's [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [dependency guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies) for current account and repository requirements.
