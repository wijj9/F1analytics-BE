# F1 Analytics BE

This directory contains the backend components for the F1 Analytics web application.

## Components

*   **`main.py`**: A FastAPI application that serves Formula 1 data. It reads pre-processed data from the `data_cache` directory and can also fetch live data using the FastF1 library.
*   **`processor.py`**: A Python script that uses the FastF1 library to fetch historical F1 data (schedules, results, standings, lap times, etc.), processes it, and saves it into JSON files within the `data_cache` directory. This pre-processing step speeds up API responses for historical data.
*   **`data_cache/`**: Directory where the `processor.py` script stores the processed JSON data files, organized by year.
*   **`cache/`**: Directory used by the FastF1 library to cache raw API responses, reducing redundant external requests.
*   **`.env`**: Environment variables file. Should contain `F1ANALYTICS_API_KEY` for securing the API endpoints and optionally `FRONTEND_URL` for CORS configuration.
*   **`requirements.txt`**: (Optional - If you create one) Lists Python dependencies.

## Setup & Usage

1.  **Dependencies:** Ensure you have Python installed. It's recommended to use a virtual environment. Install dependencies (primarily `fastapi`, `uvicorn`, `fastf1`, `pandas`, `numpy`, `python-dotenv`):
    ```bash
    # Install core dependencies + FastF1 with plotting extras
    pip install fastapi uvicorn "fastf1[full]" pandas numpy python-dotenv requests requests_cache
    # Optional: Install Gunicorn for production deployment
    pip install gunicorn
    # Or install from requirements.txt if provided
    # pip install -r requirements.txt
    ```
2.  **Environment Variables:** Create a `.env` file in this directory and add your desired `F1ANALYTICS_API_KEY`.
    ```
    F1ANALYTICS_API_KEY=your_secret_api_key_here
    FRONTEND_URL=http://localhost:8080 # Or your frontend's URL
    ```
3.  **Run Data Processor:** Execute the processor script to fetch and cache historical data. This can take some time initially.
    ```bash
    python processor.py
    ```
4.  **Run API Server:** Start the FastAPI server using Uvicorn.
    ```bash
    # For development (with auto-reload)
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```
    The API will then be accessible, typically at `http://localhost:8000`.

    **Alternative (for Production):**
    For production deployments, using Gunicorn with Uvicorn workers is often recommended for better process management.
    ```bash
    # Example: Run with 4 worker processes
    gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000 -w 4
    ```
    *   `-k uvicorn.workers.UvicornWorker`: Specifies the Uvicorn worker class for ASGI compatibility.
    *   `--bind 0.0.0.0:8000`: Sets the address and port to listen on.
    *   `-w 4`: (Optional) Specifies the number of worker processes (adjust based on your server resources).
