import os
import json
from pathlib import Path  # Keep pathlib.Path for filesystem operations
import fastapi  # Import the full fastapi module
import stripe
from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.requests import Request  # Import Request specifically from fastapi.requests
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
import numpy as np  # Keep numpy for NaN replacement if needed by chance
from dotenv import load_dotenv
import fastf1 as ff1
import pandas as pd  # Import fastf1 for schedule

# Re-import data_processing as it's needed again
from api import data_processing
import time  # Import time for logging

from dotenv import load_dotenv

from fastapi import APIRouter, Request

load_dotenv()

from supabase import create_client, Client


stripe.api_key = os.getenv("STRIPE_SECRET_KEY")  # Set this in your .env or deployment environment

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_PRIVATE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "data-cache")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Configuration ---
load_dotenv()
script_dir = Path(__file__).resolve().parent  # Get script directory
DATA_CACHE_PATH = script_dir / "data_cache"  # Root for our processed JSON
STATIC_DATA_PATH = script_dir / "static_data"  # Root for static JSON files
FASTF1_CACHE_PATH = os.getenv('FASTF1_CACHE_PATH', script_dir / 'cache')  # Needed for ff1 schedule
API_KEY_NAME = "X-API-Key"  # Standard header name for API keys
API_KEY = os.getenv("F1ANALYTICS_API_KEY")  # Read API key from environment variable
if not API_KEY:
    print("WARNING: F1ANALYTICS_API_KEY environment variable not set. API will be unsecured.")
    # Consider raising an error here if security is mandatory:
    # raise ValueError("F1ANALYTICS_API_KEY environment variable is required for security.")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_PRIVATE_KEY"))

# Ensure FastF1 cache is enabled if not already by processor
if not Path(FASTF1_CACHE_PATH).exists():
    os.makedirs(FASTF1_CACHE_PATH)
try:
    ff1.Cache.enable_cache(FASTF1_CACHE_PATH)
except Exception as e:
    print(f"Warning: Could not enable FastF1 cache in main.py: {e}")

app = FastAPI(
    title="F1 Analytics Backend API",
    description="API to serve pre-processed and live-processed Formula 1 data.",
    version="0.4.0",  # Bump version for new features
)

# --- CORS Configuration ---
FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:8080')
origins = [FRONTEND_URL,
        "https://f1analytics-git-main-wijj9s-projects.vercel.app",
        "https://f1analytics-two.vercel.app",
]
print(f"Allowing CORS origins: {origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Security Dependency ---
async def get_api_key(key: str = Depends(api_key_header)):
    """API key check disabled (for dev or open public API)"""
    print("API key check skipped. All requests are allowed.")
    return key


def read_json_cache(file_path: Path):
    """Reads JSON data from a cache file."""
    if not file_path.is_file():
        print(f"Cache file not found: {file_path}")
        return None
    try:
        with open(file_path, 'r') as f:
            # Handle potential NaN values during load if processor missed them
            return json.load(f, parse_constant=lambda x: None if x == 'NaN' else x)
    except Exception as e:
        print(f"Error reading cache file {file_path}: {e}")
        # Let the endpoint handler raise HTTPException
        return None


def read_json_from_supabase(bucket: str, path: str):
    try:
        response = supabase.storage.from_(bucket).download(path)
        content = response.decode("utf-8")
        return json.loads(content)
    except Exception as e:
        print(f"Error reading from Supabase ({bucket}/{path}): {e}")
        return None


@app.get("/")
async def read_root():
    # This root endpoint might not need protection
    return {"message": "Welcome to the F1 Analytics Backend API"}


@app.post("/api/create-checkout-session")
async def create_checkout_session(request: Request):
    try:
        data = await request.json()
        print(f"Creating checkout session for request: {data}")

        email = data.get("email")
        if not email or email == "Email":  # Handle the hardcoded "Email" case
            print("Error: No valid email provided in request")
            raise HTTPException(status_code=400, detail="A valid email is required")

        print(f"Creating Stripe checkout session for email: {email}")

        # Use the correct frontend URL from environment
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:8080")

        session = stripe.checkout.Session.create(
            payment_method_types=[
                "card",
                "klarna",
                "paypal",
            ],
            line_items=[{
                "price_data": {
                    "currency": "eur",
                    "product_data": {
                        "name": "F1Analytics Premium Access"
                    },
                    "unit_amount": 500,  # €5.00 (amount in cents)
                },
                "quantity": 1,
            }],
            mode="payment",
            customer_email=email,
            locale="en",
            success_url=f"{frontend_url}/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{frontend_url}/cancel",
            metadata={
                "email": email,  # Store email in metadata for webhook processing
            }
        )

        print(f"Stripe session created successfully: {session.id}")

        # Store pending payment record in Supabase
        try:
            result = supabase.table("subscribed_users").upsert({
                "email": email,
                "is_unlocked": False,
                "stripe_session_id": session.id,  # Store session ID
            }, on_conflict="email").execute()
            print(f"Supabase record upserted for email: {email}")
        except Exception as db_error:
            print(f"Warning: Failed to store pending payment in Supabase: {db_error}")
            # Don't fail the checkout creation if DB storage fails

        return {"url": session.url}

    except stripe.error.StripeError as stripe_error:
        print(f"Stripe error creating checkout session: {stripe_error}")
        raise HTTPException(
            status_code=400,
            detail=f"Stripe error: {str(stripe_error)}"
        )
    except Exception as e:
        print(f"Unexpected error creating checkout session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create checkout session: {str(e)}"
        )


# --- Schedule Endpoint ---
@app.get("/api/schedule/{year}", dependencies=[Depends(get_api_key)])
async def get_schedule(year: int):
    """ Retrieves the event schedule for a given year using FastF1. """
    print(f"Received request for schedule: {year}")
    try:
        # Fetch schedule directly using FastF1 (caching handled by FastF1)
        schedule_df = ff1.get_event_schedule(year, include_testing=False)

        # Define date columns to convert
        date_columns = ['EventDate', 'Session1Date', 'Session2Date', 'Session3Date', 'Session4Date', 'Session5Date']

        # Convert Timestamp columns to ISO format strings safely
        for col in date_columns:
            if col in schedule_df.columns:
                # Apply conversion only to valid Timestamps, keep NaT/None as None
                schedule_df[col] = schedule_df[col].apply(
                    lambda x: x.isoformat() if pd.notna(x) and isinstance(x, pd.Timestamp) else None)

        # Convert the DataFrame to a list of dictionaries
        # Note: NaT/None values handled by the apply function above should become null in JSON
        schedule_dict = schedule_df.to_dict(orient='records')
        print(f"Returning schedule for {year} with {len(schedule_dict)} events.")
        return schedule_dict
    except Exception as e:
        print(f"Error fetching schedule for {year}: {e}")
        # Use built-in Exception as per custom instructions
        raise HTTPException(status_code=500, detail=f"Failed to fetch schedule: {e}")


# --- Live Processing Endpoints ---

@app.get("/api/session/drivers", dependencies=[Depends(get_api_key)])
async def get_session_drivers(
        year: int = Query(..., description="Year of the season", example=2023),
        event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
        session: str = Query(..., description="Session type (e.g., R, Q, S, FP1, FP2, FP3)")
):
    """ Retrieves a list of drivers (code, name, team) who participated in a session from Supabase. """
    print(f"Received request for session drivers: {year}, {event}, {session}")
    try:
        if isinstance(event, str) and not event.isdigit():
            event_slug_corrected = event.replace(' ', '_').replace('-', '_').lower()
        else:
            event_slug_corrected = f"round_{event}".lower()

        supabase_path = f"data/{year}/charts/{event_slug_corrected}_{session.upper()}_drivers.json"
        print(f"Attempting to read Supabase file: {supabase_path}")
        cached_drivers = read_json_from_supabase(SUPABASE_BUCKET, supabase_path)

        if cached_drivers is not None:
            print(f"Returning cached session drivers for {year} {event} {session}.")
            return cached_drivers

        print(f"Cache miss for session drivers: {supabase_path}")
        raise HTTPException(status_code=404,
                            detail=f"Session drivers data not available for {year} {event} {session}.")
    except Exception as e:
        print(f"Error fetching session drivers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch session drivers: {e}")


@app.get("/api/laps/driver", dependencies=[Depends(get_api_key)])
async def get_driver_lap_numbers(
        year: int = Query(..., description="Year of the season", example=2023),
        event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
        session: str = Query(..., description="Session type"),
        driver: str = Query(..., min_length=3, max_length=3, description="3-letter driver code")
):
    """ Retrieves a list of valid lap numbers for a specific driver in a session. """
    print(f"Received request for lap numbers: {year}, {event}, {session}, driver={driver}")
    try:
        lap_numbers = data_processing.fetch_driver_lap_numbers(year, event, session, driver)
        if lap_numbers is None:  # Should return [] if no laps, None might indicate error upstream
            raise HTTPException(status_code=404, detail="Could not retrieve lap numbers.")
        return {"laps": lap_numbers}  # Return as a JSON object with a 'laps' key
    except Exception as e:
        print(f"Error fetching lap numbers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch lap numbers: {e}")


@app.get("/api/laptimes", dependencies=[Depends(get_api_key)])
async def get_lap_times(
        year: int = Query(..., description="Year of the season", example=2023),
        event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
        session: str = Query(..., description="Session type"),
        drivers: list[str] = Query(..., min_length=1, max_length=5, description="List of 1 to 5 driver codes")
):
    """ Retrieves and compares lap times for one to five drivers from Supabase. """
    print(f"Received request for laptimes: {year}, {event}, {session}, drivers={drivers}")
    if not (1 <= len(drivers) <= 5):
        raise HTTPException(status_code=400, detail="Please provide 1 to 5 driver codes.")
    try:
        if isinstance(event, str) and not event.isdigit():
            event_slug_corrected = event.replace(' ', '_').replace('-', '_').lower()
        else:
            event_slug_corrected = f"round_{event}".lower()

        supabase_path = f"data/{year}/charts/{event_slug_corrected}_{session.upper()}_laptimes.json"
        print(f"Attempting to read Supabase file: {supabase_path}")
        cached_laptimes = read_json_from_supabase(SUPABASE_BUCKET, supabase_path)

        if cached_laptimes is not None:
            filtered_data = [
                {
                    'LapNumber': lap['LapNumber'],
                    **{drv: lap.get(drv) for drv in drivers if drv in lap}
                } for lap in cached_laptimes
            ]

            if any(all(drv in lap for drv in drivers) for lap in filtered_data):
                print(f"Returning cached lap times for {drivers} in {year} {event} {session}.")
                return filtered_data

            if not filtered_data or not any(any(drv in lap for drv in drivers) for lap in filtered_data):
                print(f"Cache hit, but requested drivers {drivers} not found in {supabase_path}")
                raise HTTPException(status_code=404,
                                    detail=f"Lap time data for requested drivers not found in cache for {year} {event} {session}.")
            else:
                print(f"Returning potentially partially filtered cached lap times for {drivers} in {year} {event} {session}.")
                return filtered_data

        print(f"Cache miss for lap times: {supabase_path}")
        raise HTTPException(status_code=404,
                            detail=f"Lap time data not available for {year} {event} {session}.")
    except Exception as e:
        print(f"Error fetching lap times: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch lap times: {e}")


@app.get("/api/telemetry/speed", dependencies=[Depends(get_api_key)])
async def get_telemetry_speed(
        year: int = Query(..., description="Year of the season", example=2023),
        event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
        session: str = Query(..., description="Session type"),
        driver: str = Query(..., min_length=3, max_length=3, description="3-letter driver code"),
        lap: str = Query(default="fastest", description="Lap number (integer) or 'fastest'")  # Updated description
):
    """ Retrieves speed telemetry data for a specific driver lap. """
    print(f"Received request for speed telemetry: {year}, {event}, {session}, {driver}, lap={lap}")
    # NOTE: Telemetry is usually NOT pre-processed due to size. Fetch live.
    try:
        # data_processing needs update to handle integer lap numbers
        speed_data_df = data_processing.fetch_and_process_speed_trace(year, event, session, driver, lap)
        if speed_data_df is None or speed_data_df.empty:
            raise HTTPException(status_code=404, detail="Speed telemetry data not found.")
        result_json = speed_data_df.to_dict(orient='records')
        return result_json
    except ValueError as ve:  # Catch potential ValueError from int conversion
        print(f"Invalid lap parameter: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid lap parameter: {lap}. Must be 'fastest' or an integer.")
    except Exception as e:
        print(f"Error fetching speed telemetry: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch speed telemetry: {e}")


@app.get("/api/telemetry/gear", dependencies=[Depends(get_api_key)])
async def get_telemetry_gear(
        year: int = Query(..., description="Year of the season", example=2023),
        event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
        session: str = Query(..., description="Session type"),
        driver: str = Query(..., min_length=3, max_length=3, description="3-letter driver code"),
        lap: str = Query(default="fastest", description="Lap number (integer) or 'fastest'")  # Updated description
):
    """ Retrieves gear telemetry data (X, Y, nGear) for a specific driver lap. """
    print(f"Received request for gear telemetry: {year}, {event}, {session}, {driver}, lap={lap}")
    # NOTE: Telemetry is usually NOT pre-processed due to size. Fetch live.
    try:
        # data_processing needs update to handle integer lap numbers
        gear_data_df = data_processing.fetch_and_process_gear_map(year, event, session, driver, lap)
        if gear_data_df is None or gear_data_df.empty:
            raise HTTPException(status_code=404, detail="Gear telemetry data not found.")
        result_json = gear_data_df.to_dict(orient='records')
        return result_json
    except ValueError as ve:  # Catch potential ValueError from int conversion
        print(f"Invalid lap parameter: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid lap parameter: {lap}. Must be 'fastest' or an integer.")
    except Exception as e:
        print(f"Error fetching gear telemetry: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch gear telemetry: {e}")


@app.get("/api/telemetry/steering", dependencies=[Depends(get_api_key)])
async def get_telemetry_steering(
        year: int = Query(..., description="Year of the season", example=2023),
        event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
        session: str = Query(..., description="Session type"),
        driver: str = Query(..., min_length=3, max_length=3, description="3-letter driver code"),
        lap: str = Query(default="fastest", description="Lap number (integer) or 'fastest'")
):
    """ Retrieves steering telemetry data for a specific driver lap. """
    print(f"Received request for steering telemetry: {year}, {event}, {session}, {driver}, lap={lap}")
    try:
        steering_data_df = data_processing.fetch_and_process_steering(year, event, session, driver, lap)
        if steering_data_df is None or steering_data_df.empty:
            raise HTTPException(status_code=404, detail="Steering telemetry data not found.")
        result_json = steering_data_df.to_dict(orient='records')
        return result_json
    except ValueError as ve:
        print(f"Invalid lap parameter: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid lap parameter: {lap}. Must be 'fastest' or an integer.")
    except Exception as e:
        print(f"Error fetching steering telemetry: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch steering telemetry: {e}")


@app.get("/api/telemetry/throttle", dependencies=[Depends(get_api_key)])
async def get_telemetry_throttle(
        year: int = Query(..., description="Year of the season", example=2023),
        event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
        session: str = Query(..., description="Session type"),
        driver: str = Query(..., min_length=3, max_length=3, description="3-letter driver code"),
        lap: str = Query(default="fastest", description="Lap number (integer) or 'fastest'")
):
    """ Retrieves throttle telemetry data for a specific driver lap. """
    print(f"Received request for throttle telemetry: {year}, {event}, {session}, {driver}, lap={lap}")
    try:
        throttle_data_df = data_processing.fetch_and_process_throttle(year, event, session, driver, lap)
        if throttle_data_df is None or throttle_data_df.empty:
            raise HTTPException(status_code=404, detail="Throttle telemetry data not found.")
        result_json = throttle_data_df.to_dict(orient='records')
        return result_json
    except ValueError as ve:
        print(f"Invalid lap parameter: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid lap parameter: {lap}. Must be 'fastest' or an integer.")
    except Exception as e:
        print(f"Error fetching throttle telemetry: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch throttle telemetry: {e}")


@app.get("/api/telemetry/brake", dependencies=[Depends(get_api_key)])
async def get_telemetry_brake(
        year: int = Query(..., description="Year of the season", example=2023),
        event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
        session: str = Query(..., description="Session type"),
        driver: str = Query(..., min_length=3, max_length=3, description="3-letter driver code"),
        lap: str = Query(default="fastest", description="Lap number (integer) or 'fastest'")
):
    """ Retrieves brake telemetry data for a specific driver lap. """
    print(f"Received request for brake telemetry: {year}, {event}, {session}, {driver}, lap={lap}")
    try:
        brake_data_df = data_processing.fetch_and_process_brake(year, event, session, driver, lap)
        if brake_data_df is None or brake_data_df.empty:
            raise HTTPException(status_code=404, detail="Brake telemetry data not found.")
        result_json = brake_data_df.to_dict(orient='records')
        return result_json
    except ValueError as ve:
        print(f"Invalid lap parameter: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid lap parameter: {lap}. Must be 'fastest' or an integer.")
    except Exception as e:
        print(f"Error fetching brake telemetry: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch brake telemetry: {e}")


@app.get("/api/telemetry/rpm", dependencies=[Depends(get_api_key)])
async def get_telemetry_rpm(
        year: int = Query(..., description="Year of the season", example=2023),
        event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
        session: str = Query(..., description="Session type"),
        driver: str = Query(..., min_length=3, max_length=3, description="3-letter driver code"),
        lap: str = Query(default="fastest", description="Lap number (integer) or 'fastest'")
):
    """ Retrieves RPM telemetry data for a specific driver lap. """
    print(f"Received request for RPM telemetry: {year}, {event}, {session}, {driver}, lap={lap}")
    try:
        rpm_data_df = data_processing.fetch_and_process_rpm(year, event, session, driver, lap)
        if rpm_data_df is None or rpm_data_df.empty:
            raise HTTPException(status_code=404, detail="RPM telemetry data not found.")
        result_json = rpm_data_df.to_dict(orient='records')
        return result_json
    except ValueError as ve:
        print(f"Invalid lap parameter: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid lap parameter: {lap}. Must be 'fastest' or an integer.")
    except Exception as e:
        print(f"Error fetching RPM telemetry: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch RPM telemetry: {e}")


@app.get("/api/telemetry/drs", dependencies=[Depends(get_api_key)])
async def get_telemetry_drs(
        year: int = Query(..., description="Year of the season", example=2023),
        event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
        session: str = Query(..., description="Session type"),
        driver: str = Query(..., min_length=3, max_length=3, description="3-letter driver code"),
        lap: str = Query(default="fastest", description="Lap number (integer) or 'fastest'")
):
    """ Retrieves DRS telemetry data for a specific driver lap. """
    print(f"Received request for DRS telemetry: {year}, {event}, {session}, {driver}, lap={lap}")
    try:
        drs_data_df = data_processing.fetch_and_process_drs(year, event, session, driver, lap)
        if drs_data_df is None or drs_data_df.empty:
            raise HTTPException(status_code=404, detail="DRS telemetry data not found.")
        result_json = drs_data_df.to_dict(orient='records')
        return result_json
    except ValueError as ve:
        print(f"Invalid lap parameter: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid lap parameter: {lap}. Must be 'fastest' or an integer.")
    except Exception as e:
        print(f"Error fetching DRS telemetry: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch DRS telemetry: {e}")


@app.get("/api/comparison/sectors", dependencies=[Depends(get_api_key)])
async def get_sector_comparison(
        year: int = Query(..., description="Year of the season", example=2023),
        event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
        session: str = Query(..., description="Session type"),
        driver1: str = Query(..., min_length=3, max_length=3, description="3-letter code for driver 1"),
        driver2: str = Query(..., min_length=3, max_length=3, description="3-letter code for driver 2"),
        lap1: str = Query(default="fastest", description="Lap identifier for driver 1 (number or 'fastest')"),
        # Add lap1 param
        lap2: str = Query(default="fastest", description="Lap identifier for driver 2 (number or 'fastest')")
        # Add lap2 param
):
    """ Retrieves sector/segment comparison data between two drivers for specific laps, including SVG paths. """
    print(
        f"Received request for sector comparison: {year}, {event}, {session}, {driver1} (Lap {lap1}) vs {driver2} (Lap {lap2})")  # Update log
    if driver1 == driver2:
        raise HTTPException(status_code=400, detail="Please select two different drivers.")
    # NOTE: This is live processing, not typically cached.
    try:
        # Directly call the processing function, passing lap identifiers
        # Reverting function call and params to previous state for debugging
        comparison_data = data_processing.fetch_and_process_sector_comparison(
            year, event, session, driver1, driver2, lap1_identifier=lap1, lap2_identifier=lap2
        )
        if comparison_data is None:
            # Revert error message as well
            raise HTTPException(status_code=404,
                                detail=f"Sector comparison data could not be generated for the specified laps (Lap {lap1} vs Lap {lap2}). Check if laps exist and have telemetry.")
        return comparison_data
    # Revert exception handling block as well
    except Exception as e:
        print(f"Error fetching sector comparison: {e}")
        # Use built-in Exception as per custom instructions
        raise HTTPException(status_code=500, detail=f"Failed to fetch sector comparison: {e}")


@app.get("/api/strategy", dependencies=[Depends(get_api_key)])
async def get_tire_strategy(
        year: int = Query(..., description="Year of the season", example=2023),
        event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
        session: str = Query(..., description="Session type")
):
    """ Retrieves tire stint data for all drivers in a session. """
    print(f"Received request for tire strategy: {year}, {event}, {session}")
    try:
        strategy_data = data_processing.fetch_and_process_tire_strategy(year, event, session)
        if strategy_data is None:
            raise HTTPException(status_code=404, detail="Tire strategy data not available or failed to process.")
        return strategy_data
    except Exception as e:
        print(f"Error fetching tire strategy: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch tire strategy: {e}")


@app.get("/api/incidents", dependencies=[Depends(get_api_key)])
async def get_session_incidents_and_results(
        year: int = Query(..., description="Year of the season"),
        event: str = Query(..., description="Event name or Round Number"),
        session: str = Query(..., description="Session type (R, Q, S, etc.)")
):
    """ Fetches incident messages AND results (official or provisional) for a session. """
    start_time = time.time()
    print(f"Received request for incidents & results: {year}, {event}, {session}")
    try:
        # Call the updated function that returns both incidents and results
        incidents_list, results_df = data_processing.fetch_session_incidents_and_results(year, event, session)

        # Handle potential None DataFrame (if processing failed entirely)
        results_list = []
        if results_df is not None and not results_df.empty:
            # Convert DataFrame to list of dicts, replacing NaN/NaT with None for JSON
            results_df_processed = results_df.replace({pd.NA: None, np.nan: None})
            # Convert Timestamps/Timedeltas to string if necessary
            if 'Time' in results_df_processed.columns:
                results_df_processed['Time'] = results_df_processed['Time'].astype(str)

            results_list = results_df_processed.to_dict(orient='records')
        elif results_df is not None:  # It's an empty DataFrame
            print("Results DataFrame is empty, returning empty list for results.")
        else:
            print("Results DataFrame was None, returning empty list for results.")
            # Optionally, could raise 404 if results are critical and missing?

        # incidents_list should already be a list[dict]
        if incidents_list is None:
            print("Incidents list was None, returning empty list.")
            incidents_list = []

        end_time = time.time()
        print(
            f"Incidents & results request processed in {end_time - start_time:.2f} seconds. Returning {len(incidents_list)} incidents, {len(results_list)} results.")

        # Return both incidents and results in a structured response
        return {
            "incidents": incidents_list,
            "results": results_list
        }

    except ff1.ErgastUnavailableError as e:
        print(f"ErgastUnavailableError fetching incidents/results: {e}")
        # Maybe return a specific error structure?
        raise HTTPException(status_code=503, detail=f"Official data source (Ergast) is unavailable: {e}")
    except ff1.FastF1Error as f1_error:
        print(f"FastF1Error fetching incidents/results: {f1_error}")
        raise HTTPException(status_code=500, detail=f"Internal FastF1 error: {f1_error}")
    except Exception as e:
        print(f"Unexpected error fetching incidents/results: {e}")
        import traceback
        traceback.print_exc()  # Log full traceback for unexpected errors
        raise HTTPException(status_code=500, detail=f"Failed to fetch incidents and results: {e}")


@app.get("/api/circuit-comparison", dependencies=[Depends(get_api_key)])
async def get_circuit_comparison(
        year: int = Query(..., description="Year of the season", example=2023),
        event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
        session: str = Query(..., description="Session type"),
        driver1: str = Query(..., min_length=3, max_length=3, description="3-letter driver code 1"),
        driver2: str = Query(..., min_length=3, max_length=3, description="3-letter driver code 2"),
        lap1: str = Query(default="fastest", description="Lap identifier for driver 1 (number or 'fastest')"),
        lap2: str = Query(default="fastest", description="Lap identifier for driver 2 (number or 'fastest')")
):
    """ Retrieves sector comparison data and SVG paths for two drivers on specific laps. """
    print(
        f"Received request for circuit comparison: {year}, {event}, {session}, {driver1} (Lap {lap1}) vs {driver2} (Lap {lap2})")
    if driver1 == driver2 and lap1 == lap2:
        raise HTTPException(status_code=400, detail="Cannot compare the same driver on the same lap.")
    try:
        comparison_data = data_processing.fetch_and_process_sector_comparison(year, event, session, driver1, driver2,
                                                                              lap1, lap2)
        if comparison_data is None:
            raise HTTPException(status_code=404,
                                detail="Circuit comparison data not available for the selected drivers/laps.")
        return comparison_data
    except Exception as e:
        print(f"Error fetching circuit comparison: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch circuit comparison: {e}")


@app.get("/api/lapdata/positions", dependencies=[Depends(get_api_key)])
async def get_lap_positions(
        year: int = Query(..., description="Year of the season", example=2023),
        event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
        session: str = Query(..., description="Session type (R or S)")
):
    """ Retrieves lap-by-lap position data for all drivers in a race or sprint session from Supabase. """
    print(f"Received request for cached lap positions: {year}, {event}, {session}")
    if session not in ['R', 'S']:
        raise HTTPException(status_code=400,
                            detail="Position data is only available for Race (R) or Sprint (S) sessions.")
    try:
        if isinstance(event, str) and not event.isdigit():
            event_slug_corrected = event.replace(' ', '_').replace('-', '_').lower()
        else:
            event_slug_corrected = f"round_{event}".lower()

        supabase_path = f"data/{year}/charts/{event_slug_corrected}_{session.upper()}_positions.json"
        print(f"Attempting to read Supabase file: {supabase_path}")
        positions_data = read_json_from_supabase(SUPABASE_BUCKET, supabase_path)

        if positions_data is None:
            raise HTTPException(status_code=404,
                                detail=f"Lap position data not available for {year} {event} {session}.")

        print(f"Returning cached lap positions for {year} {event} {session}.")
        return positions_data
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Unexpected error: {e}"); raise HTTPException(status_code=500)


# --- Standings & Results Endpoints (Read from Cache) ---

@app.get("/api/standings/drivers", dependencies=[Depends(get_api_key)])
async def get_driver_standings_api(year: int = Query(...)):
    """ Retrieves pre-calculated driver standings for a given year from cache. """
    start_time = time.time()
    print(f"{start_time:.2f} - REQ START: Driver Standings {year}")
    try:
        supabase_path = f"data/{year}/standings/standings.json"
        standings_data = read_json_from_supabase(SUPABASE_BUCKET, supabase_path)
        if standings_data is None or 'drivers' not in standings_data:
            print(f"{time.time():.2f} - REQ ERROR: Driver Standings {year} - Not Found")
            raise HTTPException(status_code=404,
                                detail=f"Driver standings not available for {year}.")
        result = standings_data['drivers']
        print(f"{time.time():.2f} - REQ END: Driver Standings {year} ({time.time() - start_time:.3f}s)")
        return result
    except HTTPException as http_exc:
        print(f"{time.time():.2f} - REQ HTTP EXC: Driver Standings {year} ({http_exc.status_code})")
        raise http_exc
    except Exception as e:
        print(f"{time.time():.2f} - REQ UNEXP EXC: Driver Standings {year} - {e}")
        raise HTTPException(status_code=500)


@app.get("/api/standings/teams", dependencies=[Depends(get_api_key)])
async def get_team_standings_api(year: int = Query(...)):
    """ Retrieves pre-calculated constructor standings for a given year from cache. """
    start_time = time.time()
    print(f"{start_time:.2f} - REQ START: Team Standings {year}")
    try:
        supabase_path = f"data/{year}/standings/standings.json"
        standings_data = read_json_from_supabase(SUPABASE_BUCKET, supabase_path)
        if standings_data is None or 'teams' not in standings_data:
            print(f"{time.time():.2f} - REQ ERROR: Team Standings {year} - Not Found")
            raise HTTPException(status_code=404,
                                detail=f"Team standings not available for {year}.")
        result = standings_data['teams']
        print(f"{time.time():.2f} - REQ END: Team Standings {year} ({time.time() - start_time:.3f}s)")
        return result
    except HTTPException as http_exc:
        print(f"{time.time():.2f} - REQ HTTP EXC: Team Standings {year} ({http_exc.status_code})")
        raise http_exc
    except Exception as e:
        print(f"{time.time():.2f} - REQ UNEXP EXC: Team Standings {year} - {e}")
        raise HTTPException(status_code=500)


@app.get("/api/results/races", dependencies=[Depends(get_api_key)])
async def get_race_results_summary_api(year: int = Query(...)):
    """ Retrieves summary of race results (winners) for a given year from cache. """
    start_time = time.time()
    print(f"{start_time:.2f} - REQ START: Race Results Summary {year}")
    try:
        supabase_path = f"data/{year}/race_results.json"
        results_data = read_json_from_supabase(SUPABASE_BUCKET, supabase_path)
        if results_data is None:
            print(f"{time.time():.2f} - REQ ERROR: Race Results Summary {year} - Not Found")
            raise HTTPException(status_code=404,
                                detail=f"Race results summary not available for {year}.")
        print(f"{time.time():.2f} - REQ END: Race Results Summary {year} ({time.time() - start_time:.3f}s)")
        return results_data
    except HTTPException as http_exc:
        print(f"{time.time():.2f} - REQ HTTP EXC: Race Results Summary {year} ({http_exc.status_code})")
        raise http_exc
    except Exception as e:
        print(f"{time.time():.2f} - REQ UNEXP EXC: Race Results Summary {year} - {e}")
        raise HTTPException(status_code=500)


@app.get("/api/sessions", dependencies=[Depends(get_api_key)])
async def get_available_sessions(
        year: int = Query(..., description="Year of the season"),
        event: str = Query(..., description="Event name or Round Number")
):
    """ Retrieves available processed session files (including segments) for a given event. """
    print(f"Received request for available sessions: {year}, {event}")
    try:
        # --- Determine event_slug (consistent with processor, ensure lowercase for file ops) ---
        event_slug_raw = None
        try:
            # Try converting event to int first (for round number)
            round_num = int(event)
            schedule = ff1.get_event_schedule(year, include_testing=False)
            event_row = schedule[schedule['RoundNumber'] == round_num]
            if not event_row.empty:
                event_slug_raw = event_row['EventName'].iloc[0].replace(' ', '_')
            else:
                raise HTTPException(status_code=404, detail=f"Event round {event} not found for {year}")
        except ValueError:
            # If not an integer, assume it's an event name
            event_slug_raw = event.replace(' ', '_')
            # Optional: Verify event name exists in schedule if needed
            # schedule = ff1.get_event_schedule(year, include_testing=False)
            # if not any(schedule['EventName'] == event):
            #     raise HTTPException(status_code=404, detail=f"Event name '{event}' not found for {year}")

        if not event_slug_raw:  # Should not happen if logic above is correct
            raise HTTPException(status_code=400, detail="Could not determine event slug.")

        # Use lowercase slug for filesystem operations
        event_slug_lower = event_slug_raw.lower()

        # --- Find available session files ---
        event_results_path = f"data/{year}/races"


        available_sessions = []
        # Define the order and mapping for all possible sessions/segments
        session_order_map = {
            'FP1': 'Practice 1', 'FP2': 'Practice 2', 'FP3': 'Practice 3',
            'SQ1': 'Sprint Quali 1', 'SQ2': 'Sprint Quali 2', 'SQ3': 'Sprint Quali 3',
            'Q1': 'Qualifying 1', 'Q2': 'Qualifying 2', 'Q3': 'Qualifying 3',
            'Sprint': 'Sprint Race', 'R': 'Race',
            # Add fallbacks for parent sessions if segments aren't processed
            'SQ': 'Sprint Quali', 'Q': 'Qualifying'
        }
        # Create ordered list based on the map keys
        session_order = list(session_order_map.keys())

        # Use lowercase slug for globbing
        processed_files = set(f.name for f in event_results_path.glob(f"{event_slug_lower}_*.json"))
        print(f"Globbing for files matching: {event_slug_lower}_*.json in {event_results_path}")  # Debug log
        print(f"Found files: {processed_files}")  # Debug log

        for session_id in session_order:
            filename = f"data/{year}/races/{event_slug_lower}_{session_id}.json"
            session_data = read_json_from_supabase(SUPABASE_BUCKET, filename)
            if session_data:
                if session_id == 'Q' and any(
                        read_json_from_supabase(SUPABASE_BUCKET, f"data/{year}/races/{event_slug_lower}_q{i}.json") for
                        i in [1, 2, 3]
                ):
                    continue
                if session_id == 'SQ' and any(
                        read_json_from_supabase(SUPABASE_BUCKET, f"data/{year}/races/{event_slug_lower}_sq{i}.json") for
                        i in [1, 2, 3]
                ):
                    continue

                available_sessions.append({
                    "name": session_order_map.get(session_id, session_id),
                    "type": session_id,
                })

        print(
            f"Found available processed sessions for {year} {event_slug_lower}: {[s['type'] for s in available_sessions]}")
        available_sessions.sort(key=lambda x: session_order.index(x['type']) if x['type'] in session_order else 99)
        return available_sessions
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error fetching available sessions for {year} {event}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to fetch available sessions: {e}")


@app.get("/api/results/race/{year}/{event_slug}", dependencies=[Depends(get_api_key)])
async def get_specific_race_result_api(
        year: int,
        event_slug: str,
        session: str = Query(...)
):
    print(f"Received request for specific race result: {year}, {event_slug}, {session}")
    try:
        event_slug_lower = event_slug.lower().replace(' ', '_').replace('-', '_')
        filename = f"data/{year}/races/{event_slug_lower}_{session.upper()}.json"
        print(f"Looking for race results file: {filename}")

        results_data = read_json_from_supabase(SUPABASE_BUCKET, filename)
        if results_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Results not available yet for {year} {event_slug} {session}."
            )

        return results_data
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error fetching specific race results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch race results: {e}")


@app.get("/api/tire-strategy", dependencies=[Depends(get_api_key)])
async def get_tire_strategy(
        year: int = Query(..., description="Year of the season", example=2023),
        event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
        session: str = Query(..., description="Session type")
):
    # ... endpoint code ...
    pass


# Restore the Stint Analysis endpoint
@app.get("/api/stint-analysis", dependencies=[Depends(get_api_key)])
async def get_stint_analysis(
        year: int = Query(..., description="Year of the season", example=2023),
        event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
        session: str = Query(..., description="Session type")
):
    """ Retrieves detailed stint analysis data including lap times for each stint. """
    print(f"Received request for stint analysis: {year}, {event}, {session}")
    try:
        stint_data = data_processing.fetch_and_process_stint_analysis(year, event, session)
        if stint_data is None:
            raise HTTPException(status_code=404, detail="Stint analysis data not found or session invalid.")
        if not stint_data:
            print(f"Returning empty list for stint analysis: {year} {event} {session}")
            # Return 200 with empty list if processing was successful but yielded no stints
        return stint_data
    except Exception as e:
        print(f"Error fetching stint analysis: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to fetch stint analysis: {e}")


@app.get("/api/test/supabase-read")
async def test_supabase_read():
    """Test if Supabase download works for a known file path."""
    test_path = "data/2024/race_results.json"  # Update this path to a real file you know exists in Supabase

    data = read_json_from_supabase(SUPABASE_BUCKET, test_path)

    if data is None:
        raise HTTPException(status_code=404, detail=f"File not found or failed to decode at {test_path}")

    return {"status": "success", "preview": data[:3] if isinstance(data, list) else data}



@app.get("/api/check-subscription")
async def check_subscription(email: str):
    """Check if a user has an active subscription"""
    try:
        result = supabase.table("subscribed_users").select("is_active, payment_status").eq("email", email).execute()
        if result.data and len(result.data) > 0:
            user_data = result.data[0]
            return {
                "isSubscribed": user_data.get("is_active", False),
                "paymentStatus": user_data.get("payment_status", "none")
            }
        return {"isSubscribed": False, "paymentStatus": "none"}
    except Exception as e:
        print(f"Error checking subscription for {email}: {e}")
        return {"isSubscribed": False, "paymentStatus": "error"}


@app.post("/api/create-payment-intent")
async def create_payment_intent(request: Request):
    data = await request.json()
    try:
        intent = stripe.PaymentIntent.create(
            amount=data["amount"],  # amount in cents
            currency="usd",
            # You can add more parameters as needed
        )
        return {"clientSecret": intent.client_secret}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
        print(f"Received Stripe webhook event: {event['type']}")
    except Exception as e:
        print(f"Webhook signature verification failed: {e}")
        raise HTTPException(status_code=400, detail=f"Webhook signature verification failed: {str(e)}")

    # Handle successful payment completion
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        customer_email = session.get("customer_email")
        session_id = session.get("id")

        print(f"Processing successful payment for email: {customer_email}, session: {session_id}")

        if not customer_email:
            print("Warning: No customer email found in webhook session data")
            return {"status": "success", "message": "No customer email found"}

        try:
            # Update user as active subscriber with more detailed tracking
            result = supabase.table("subscribed_users").update({
                "is_active": True,
                "stripe_customer_id": session.get("customer"),
                "stripe_session_id": session_id,
                "payment_status": "completed",
                "payment_amount": session.get("amount_total", 500),  # Amount in cents
                "payment_currency": session.get("currency", "eur"),
                "payment_date": "now()",
                "updated_at": "now()"
            }).eq("email", customer_email).execute()

            print(f"Successfully updated subscription status for {customer_email}")

            # Optional: Log the payment in a separate payments table for better tracking
            try:
                payment_record = supabase.table("payments").insert({
                    "email": customer_email,
                    "stripe_session_id": session_id,
                    "stripe_customer_id": session.get("customer"),
                    "amount": session.get("amount_total", 500),
                    "currency": session.get("currency", "eur"),
                    "payment_status": "completed",
                    "payment_method": "stripe_checkout",
                    "created_at": "now()"
                }).execute()
                print(f"Payment record created for {customer_email}")
            except Exception as payment_log_error:
                print(f"Warning: Failed to log payment record: {payment_log_error}")
                # Don't fail the webhook if payment logging fails

        except Exception as db_error:
            print(f"Error updating subscription status: {db_error}")
            # Consider whether to return error or success here
            # Returning success to avoid Stripe retrying if the payment actually succeeded
            return {"status": "error", "message": f"Database update failed: {str(db_error)}"}

    elif event["type"] == "checkout.session.expired":
        # Handle expired checkout sessions
        session = event["data"]["object"]
        customer_email = session.get("customer_email")
        session_id = session.get("id")

        print(f"Checkout session expired for email: {customer_email}, session: {session_id}")

        if customer_email:
            try:
                # Update the payment status to expired
                supabase.table("subscribed_users").update({
                    "payment_status": "expired",
                    "updated_at": "now()"
                }).eq("email", customer_email).eq("stripe_session_id", session_id).execute()
            except Exception as db_error:
                print(f"Error updating expired session status: {db_error}")

    else:
        print(f"Unhandled webhook event type: {event['type']}")

    return {"status": "success"}


@app.get("/api/verify-payment")
async def verify_payment(session_id: str):
    """Verify payment completion and update subscription status"""
    try:
        # Retrieve the session from Stripe
        session = stripe.checkout.Session.retrieve(session_id)

        if session.payment_status == "paid":
            customer_email = session.customer_email

            if customer_email:
                # Update subscription status in database
                result = supabase.table("subscribed_users").upsert({
                    "email": customer_email,
                    "is_active": True,
                    "payment_status": "completed",
                    "stripe_session_id": session_id,
                    "stripe_customer_id": session.customer,
                    "payment_amount": session.amount_total,
                    "payment_currency": session.currency,
                    "updated_at": "now()"
                }, on_conflict="email").execute()

                print(f"Payment verified and subscription activated for {customer_email}")

                return {
                    "success": True,
                    "email": customer_email,
                    "isSubscribed": True
                }

        return {"success": False, "message": "Payment not completed"}

    except Exception as e:
        print(f"Error verifying payment for session {session_id}: {e}")
        return {"success": False, "message": str(e)}
