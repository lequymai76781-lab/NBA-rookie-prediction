# NBA Rookie Prediction

FastAPI app serving `index.html` and ML prediction endpoints from `main.py`.

## Run
- Workflow `Server`: `python main.py` on port 5000 (host `0.0.0.0`).
- Deployment: autoscale, `python main.py`.

## Notes
- Models loaded from `*.pkl` at repo root.
- Reload disabled to keep PORT env honored.
