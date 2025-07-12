import fastf1 as ff1
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
import os
import time
import gc
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt # Needed for plotting utilities
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
# Removed specific FastF1 exception imports
# from fastf1.exceptions import SessionNotAvailableError, DataNotLoadedError, FastF1Error

# --- Configuration ---
FASTF1_CACHE_PATH = os.getenv('FASTF1_CACHE_PATH', './cache')
DATA_CACHE_PATH = Path("./data_cache") # Keep for processor part

if not os.path.exists(FASTF1_CACHE_PATH):
    os.makedirs(FASTF1_CACHE_PATH)
ff1.Cache.enable_cache(FASTF1_CACHE_PATH)

# --- Helper Functions ---

def map_session_identifier_for_load(session_identifier: str) -> str:
    """Maps session identifiers like Q1, Q2, Q3 to their parent (Q) for ff1.get_session."""
    if session_identifier in ['Q1', 'Q2', 'Q3']:
        return 'Q'
    elif session_identifier in ['SQ1', 'SQ2', 'SQ3']:
        return 'SQ' # Assuming Sprint Qualifying uses SQ
    # Add other mappings if needed (e.g., Sprint Shootout)
    return session_identifier # Return original if not a segment

# Helper to get team color mapping
def get_team_color_name(team_name: str | None) -> str:
    if not team_name: return 'gray'
    simple_name = team_name.lower().replace(" ", "").replace("-", "")
    if 'mclaren' in simple_name: return 'mclaren'
    if 'mercedes' in simple_name: return 'mercedes'
    if 'redbull' in simple_name: return 'redbull'
    if 'ferrari' in simple_name: return 'ferrari'
    if 'alpine' in simple_name: return 'alpine'
    if 'astonmartin' in simple_name: return 'astonmartin'
    if 'williams' in simple_name: return 'williams';
    if 'haas' in simple_name: return 'haas'
    if 'sauber' in simple_name: return 'alfaromeo'
    if 'racingbulls' in simple_name or 'alphatauri' in simple_name: return 'alphatauri'
    return 'gray'

def save_json(data, file_path: Path):
    """Saves data to a JSON file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            if isinstance(data, pd.DataFrame):
                data = data.replace({np.nan: None})
                json.dump(data.to_dict(orient='records'), f, indent=2, default=str)
            elif isinstance(data, (list, dict)):
                 json.dump(data, f, indent=2, default=str)
            else:
                 print(f" -> Unsupported data type for JSON saving: {type(data)}")
                 return
        print(f" -> Data successfully saved to {file_path}")
    except Exception as e:
        print(f" -> Error saving JSON to {file_path}: {e}")
        import traceback
        traceback.print_exc()

# --- Data Processing Functions (for Live API Endpoints) ---

def fetch_session_drivers(year: int, event: str, session_type: str) -> list[dict] | None:
    """ Fetches the list of drivers who participated in a session. """
    print(f"Fetching drivers for {year} {event} {session_type}")
    try:
        session_to_load = map_session_identifier_for_load(session_type)
        print(f" -> Mapped session type {session_type} to {session_to_load} for loading")
        session = ff1.get_session(year, event, session_to_load)
        # Removed results=True from load(), results are accessed directly via session.results
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        print("Session loaded for driver list.")

        driver_list = []
        if session.results is not None and not session.results.empty:
            driver_list = [{"code": row['Abbreviation'], "name": row['LastName'], "team": row['TeamName']}
                           for _, row in session.results.iterrows() if pd.notna(row['Abbreviation'])]
        elif session.laps is not None and not session.laps.empty:
             drivers_info = session.laps[['Driver', 'Team', 'FullName']].drop_duplicates(subset=['Driver'])
             if not drivers_info.empty:
                 driver_list = [{"code": row['Driver'], "name": row['FullName'], "team": row['Team']}
                                for _, row in drivers_info.iterrows() if pd.notna(row['Driver'])]

        if not driver_list:
             print("No drivers found in session data.")
             return None

        driver_list.sort(key=lambda x: x['code'])
        print(f"Found {len(driver_list)} drivers.")
        return driver_list
    except Exception as e: # Catch general Exception
        print(f"Error fetching session drivers: {e}")
        raise e # Re-raise for the API endpoint to handle


def fetch_and_process_laptimes_multi(year: int, event: str, session_type: str, driver_codes: list[str]) -> pd.DataFrame | None:
    """ Fetches and processes lap times for multiple drivers (2 or 3). """
    print(f"Processing lap times for {driver_codes} - {year} {event} {session_type}")
    if not (2 <= len(driver_codes) <= 3): raise ValueError("Supports 2 or 3 drivers.")
    try:
        session_to_load = map_session_identifier_for_load(session_type)
        print(f" -> Mapped session type {session_type} to {session_to_load} for loading")
        session = ff1.get_session(year, event, session_to_load)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps.pick_drivers(driver_codes).pick_accurate()
        if laps.empty: return None
        laps_copy = laps.copy()
        laps_copy['LapTimeSeconds'] = laps_copy['LapTime'].dt.total_seconds()
        laps_filtered = laps_copy[['LapNumber', 'Driver', 'LapTimeSeconds']]
        lap_comparison_df = laps_filtered.pivot_table(index='LapNumber', columns='Driver', values='LapTimeSeconds').reset_index()
        for code in driver_codes:
            if code not in lap_comparison_df.columns: lap_comparison_df[code] = pd.NA
        final_df = lap_comparison_df[['LapNumber'] + driver_codes]
        final_df = final_df.replace({np.nan: None})
        print(f"Successfully processed {len(final_df)} laps for comparison.")
        return final_df
    except Exception as e: print(f"Error processing multi-lap times: {e}"); raise e


def fetch_and_process_speed_trace(year: int, event: str, session_type: str, driver_code: str, lap_identifier: str) -> pd.DataFrame | None:
    """ Fetches telemetry for a specific lap and processes speed data. """
    print(f"Processing speed trace for {driver_code} lap {lap_identifier} - {year} {event} {session_type}")
    try:
        session_to_load = map_session_identifier_for_load(session_type)
        print(f" -> Mapped session type {session_type} to {session_to_load} for loading")
        session = ff1.get_session(year, event, session_to_load)
        session.load(laps=True, telemetry=True, weather=False, messages=False)
        # Use pick_drivers instead of pick_driver
        laps = session.laps.pick_drivers([driver_code])
        if laps.empty: return None
        target_lap = None
        if lap_identifier.lower() == 'fastest':
            target_lap = laps.pick_fastest()
            if target_lap is None or pd.isna(target_lap['LapTime']): target_lap = laps.iloc[-1] if not laps.empty else None
        else:
            try: lap_num = int(lap_identifier); lap_row = laps[laps['LapNumber'] == lap_num]; target_lap = lap_row.iloc[0] if not lap_row.empty else None
            except (ValueError, IndexError): return None
        if target_lap is None: return None
        telemetry = target_lap.get_car_data(pad=1)
        if telemetry.empty or 'Speed' not in telemetry.columns: return None
        if 'Distance' not in telemetry.columns: telemetry = telemetry.add_distance()
        speed_df = telemetry[['Distance', 'Speed']].copy().dropna(subset=['Distance', 'Speed']).replace({np.nan: None})
        print(f"Successfully processed speed trace for lap {getattr(target_lap, 'LapNumber', 'N/A')}. Records: {len(speed_df)}")
        return speed_df
    except Exception as e: print(f"Error processing speed trace: {e}"); raise e


def fetch_and_process_gear_map(year: int, event: str, session_type: str, driver_code: str, lap_identifier: str) -> pd.DataFrame | None:
    """ Fetches telemetry for a specific lap and processes X, Y, Gear data. """
    print(f"Processing gear map for {driver_code} lap {lap_identifier} - {year} {event} {session_type}")
    try:
        session_to_load = map_session_identifier_for_load(session_type)
        print(f" -> Mapped session type {session_type} to {session_to_load} for loading")
        session = ff1.get_session(year, event, session_to_load)
        session.load(laps=True, telemetry=True, weather=False, messages=False)
        # Use pick_drivers instead of pick_driver
        laps = session.laps.pick_drivers([driver_code])
        if laps.empty: return None
        target_lap = None
        if lap_identifier.lower() == 'fastest':
            target_lap = laps.pick_fastest()
            if target_lap is None or pd.isna(target_lap['LapTime']): target_lap = laps.iloc[-1] if not laps.empty else None
        else:
            try: lap_num = int(lap_identifier); lap_row = laps[laps['LapNumber'] == lap_num]; target_lap = lap_row.iloc[0] if not lap_row.empty else None
            except (ValueError, IndexError): return None
        if target_lap is None: return None
        telemetry = target_lap.get_telemetry()
        required_cols = ['X', 'Y', 'nGear']
        if telemetry.empty or not all(col in telemetry.columns for col in required_cols):
             print(f"Required telemetry data (X, Y, nGear) not found for lap {getattr(target_lap, 'LapNumber', 'N/A')}.")
             return None
        gear_df = telemetry[required_cols].copy().dropna().reset_index(drop=True)
        gear_df['nGear'] = gear_df['nGear'].astype(int)
        gear_df = gear_df.replace({np.nan: None})
        print(f"Successfully processed gear map data for lap {getattr(target_lap, 'LapNumber', 'N/A')}. Records: {len(gear_df)}")
        return gear_df
    except Exception as e: print(f"Error processing gear map: {e}"); raise e


def fetch_and_process_tire_strategy(year: int, event: str, session_type: str) -> list[dict] | None:
    """ Fetches lap data and extracts tire stint information. """
    print(f"Processing tire strategy - {year} {event} {session_type}")
    try:
        # Tire strategy often relates to Race or Sprint, mapping might not be needed
        # but doesn't hurt to add for consistency if segments were ever relevant here.
        session_to_load = map_session_identifier_for_load(session_type)
        print(f" -> Mapped session type {session_type} to {session_to_load} for loading")
        session = ff1.get_session(year, event, session_to_load)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps
        if laps.empty: return None
        drivers = laps['Driver'].unique()
        if len(drivers) == 0: return None
        strategy_list = []
        for drv_code in drivers:
            try:
                # Use pick_drivers instead of pick_driver
                drv_laps = laps.pick_drivers([drv_code])
                if drv_laps.empty: continue
                stints_grouped = drv_laps.groupby("Stint")
                stint_data = []
                for stint_num, stint_laps in stints_grouped:
                    if stint_laps.empty: continue
                    compound = stint_laps["Compound"].iloc[0]
                    start_lap = stint_laps["LapNumber"].min()
                    end_lap = stint_laps["LapNumber"].max()
                    lap_count = len(stint_laps)
                    stint_data.append({"compound": compound, "startLap": int(start_lap), "endLap": int(end_lap), "lapCount": int(lap_count)})
                if stint_data:
                    stint_data.sort(key=lambda x: x['startLap'])
                    strategy_list.append({"driver": drv_code, "stints": stint_data})
            except Exception as inner_e: print(f"Error processing stints for driver {drv_code}: {inner_e}")
        print(f"Successfully processed tire strategy for {len(strategy_list)} drivers.")
        return strategy_list
    except Exception as e: print(f"Error processing tire strategy: {e}"); raise e


def fetch_driver_lap_numbers(year: int, event: str, session_type: str, driver_code: str) -> list[int] | None:
    """ Fetches valid lap numbers completed by a specific driver in a session. """
    print(f"Fetching lap numbers for {driver_code} - {year} {event} {session_type}")
    try:
        session_to_load = map_session_identifier_for_load(session_type)
        print(f" -> Mapped session type {session_type} to {session_to_load} for loading")
        session = ff1.get_session(year, event, session_to_load)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        # Use pick_drivers instead of pick_driver
        laps = session.laps.pick_drivers([driver_code]) # Removed pick_accurate() to include outlaps
        if laps.empty:
            print(f"No laps found for driver {driver_code}.")
            return [] # Return empty list if no laps
        # Get unique, sorted lap numbers and convert to standard Python int
        lap_numbers_np = sorted(laps['LapNumber'].dropna().unique().astype(int))
        lap_numbers = [int(lap) for lap in lap_numbers_np] # Convert numpy.int64 to int
        print(f"Found {len(lap_numbers)} lap numbers for {driver_code}.")
        return lap_numbers
    except Exception as e:
        print(f"Error fetching lap numbers for {driver_code}: {e}")
        raise e


def generate_svg_path(points: np.ndarray, scale: float = 1.0, offset_x: float = 0.0, offset_y: float = 0.0) -> str:
    """Generates an SVG path string from a list of (x, y) points."""
    if points.shape[0] < 2: return ""
    path_d = f"M {(points[0, 0] * scale) + offset_x},{(points[0, 1] * scale) + offset_y}"
    for i in range(1, points.shape[0]):
        path_d += f" L {(points[i, 0] * scale) + offset_x},{(points[i, 1] * scale) + offset_y}"
    return path_d

def normalize_coordinates(telemetry_df: pd.DataFrame, viewbox_width: int = 1000, viewbox_height: int = 500, padding: int = 50) -> tuple[pd.DataFrame, float, float, float]:
    """Normalizes X, Y coordinates to fit within a specified SVG viewbox."""
    x_min, x_max = telemetry_df['X'].min(), telemetry_df['X'].max()
    y_min, y_max = telemetry_df['Y'].min(), telemetry_df['Y'].max()

    range_x = x_max - x_min
    range_y = y_max - y_min

    if range_x == 0 or range_y == 0: # Avoid division by zero if track is a straight line (unlikely)
        scale = 1.0
    else:
        scale_x = (viewbox_width - 2 * padding) / range_x
        scale_y = (viewbox_height - 2 * padding) / range_y
        scale = min(scale_x, scale_y) # Use the smaller scale to fit both dimensions

    # Calculate offsets to center the track
    scaled_width = range_x * scale
    scaled_height = range_y * scale
    offset_x = padding + (viewbox_width - 2 * padding - scaled_width) / 2 - (x_min * scale)
    offset_y = padding + (viewbox_height - 2 * padding - scaled_height) / 2 - (y_min * scale)

    # Apply scaling and offset
    telemetry_df['X_norm'] = (telemetry_df['X'] * scale) + offset_x
    telemetry_df['Y_norm'] = (telemetry_df['Y'] * scale) + offset_y

    return telemetry_df, scale, offset_x, offset_y


def fetch_and_process_sector_comparison(
    year: int, event: str, session_type: str,
    driver1_code: str, driver2_code: str,
    lap1_identifier: str = 'fastest', lap2_identifier: str = 'fastest' # Add lap identifiers
) -> dict | None:
    """ Fetches telemetry for two drivers for specific laps (or fastest), compares sector times, and generates SVG paths. """
    print(f"Processing sector comparison for {driver1_code} (Lap {lap1_identifier}) vs {driver2_code} (Lap {lap2_identifier}) - {year} {event} {session_type}")
    try:
        session_to_load = map_session_identifier_for_load(session_type)
        print(f" -> Mapped session type {session_type} to {session_to_load} for loading")
        session = ff1.get_session(year, event, session_to_load)
        session.load(laps=True, telemetry=True, weather=False, messages=False)

        # Use pick_drivers instead of pick_driver
        laps_driver1 = session.laps.pick_drivers([driver1_code])
        laps_driver2 = session.laps.pick_drivers([driver2_code])

        # --- Get target lap for driver 1 ---
        if lap1_identifier.lower() == 'fastest':
            lap1 = laps_driver1.pick_fastest()
        else:
            try: lap_num = int(lap1_identifier); lap1 = laps_driver1[laps_driver1['LapNumber'] == lap_num].iloc[0] if not laps_driver1[laps_driver1['LapNumber'] == lap_num].empty else None
            except (ValueError, IndexError): lap1 = None

        # --- Get target lap for driver 2 ---
        if lap2_identifier.lower() == 'fastest':
            lap2 = laps_driver2.pick_fastest()
        else:
            try: lap_num = int(lap2_identifier); lap2 = laps_driver2[laps_driver2['LapNumber'] == lap_num].iloc[0] if not laps_driver2[laps_driver2['LapNumber'] == lap_num].empty else None
            except (ValueError, IndexError): lap2 = None

        # --- Validate laps ---
        if lap1 is None or pd.isna(lap1['LapTime']):
            print(f"Lap {lap1_identifier} data not available for driver {driver1_code}.")
            return None
        if lap2 is None or pd.isna(lap2['LapTime']):
            print(f"Lap {lap2_identifier} data not available for driver {driver2_code}.")
            return None

        # --- Get Telemetry ---
        tel1 = lap1.get_telemetry().add_distance()
        tel2 = lap2.get_telemetry().add_distance()

        # --- Validate Telemetry ---
        if tel1.empty or 'X' not in tel1.columns or 'Y' not in tel1.columns or 'Distance' not in tel1.columns or 'Time' not in tel1.columns:
             print(f"Telemetry data (X, Y, Distance, Time) missing for lap {lap1_identifier} of driver {driver1_code}.")
             return None
        if tel2.empty or 'X' not in tel2.columns or 'Y' not in tel2.columns or 'Distance' not in tel2.columns or 'Time' not in tel2.columns:
             print(f"Telemetry data (X, Y, Distance, Time) missing for lap {lap2_identifier} of driver {driver2_code}.")
             return None

        # --- Normalize Coordinates and Generate Base Layout ---
        # Use telemetry from the first driver for the base layout shape
        tel1_norm, scale, offset_x, offset_y = normalize_coordinates(tel1.copy())
        circuit_layout_svg = generate_svg_path(tel1_norm[['X_norm', 'Y_norm']].values, 1.0, 0.0, 0.0) # Already scaled

        # --- Segment the Track (Example: using corners) ---
        # This is a simplified segmentation. A more robust approach might use sectors or mini-sectors.
        # FastF1 v3+ might have better ways to get circuit info if available.
        # For now, let's use distance markers as simple segments.
        num_segments = 20 # Arbitrary number of segments
        total_distance = tel1['Distance'].max()
        segment_length = total_distance / num_segments
        segment_boundaries = np.linspace(0, total_distance, num_segments + 1)

        # --- Compare Time per Segment ---
        # Interpolate time based on distance for comparison
        time_interpolator1 = interp1d(tel1['Distance'], tel1['Time'].dt.total_seconds(), bounds_error=False, fill_value="extrapolate")
        time_interpolator2 = interp1d(tel2['Distance'], tel2['Time'].dt.total_seconds(), bounds_error=False, fill_value="extrapolate")

        track_sections = []
        for i in range(num_segments):
            start_dist = segment_boundaries[i]
            end_dist = segment_boundaries[i+1]

            # Get telemetry points within this distance segment for driver 1 (for path)
            segment_tel1 = tel1_norm[(tel1_norm['Distance'] >= start_dist) & (tel1_norm['Distance'] <= end_dist)]
            if segment_tel1.empty: continue

            # Generate SVG path for this segment
            segment_path_svg = generate_svg_path(segment_tel1[['X_norm', 'Y_norm']].values, 1.0, 0.0, 0.0) # Already scaled

            # Calculate time taken by each driver for this distance segment
            time1_start = time_interpolator1(start_dist)
            time1_end = time_interpolator1(end_dist)
            time2_start = time_interpolator2(start_dist)
            time2_end = time_interpolator2(end_dist)

            if pd.isna(time1_start) or pd.isna(time1_end) or pd.isna(time2_start) or pd.isna(time2_end):
                advantage = None # Cannot compare if interpolation failed
            else:
                delta_time1 = time1_end - time1_start
                delta_time2 = time2_end - time2_start
                # Advantage: positive means driver 1 is faster (less time), negative means driver 2 is faster
                advantage = delta_time2 - delta_time1

            track_sections.append({
                "id": f"segment_{i+1}",
                "name": f"Segment {i+1}",
                "type": "sector", # Simplified type
                "path": segment_path_svg,
                "driver1Advantage": advantage
            })

        print(f"Successfully processed sector comparison. Found {len(track_sections)} segments.")
        return {
            "sections": track_sections,
            "driver1Code": driver1_code,
            "driver2Code": driver2_code,
            "circuitLayout": circuit_layout_svg
        }

    except Exception as e:
        print(f"Error processing sector comparison: {e}")
        import traceback
        traceback.print_exc()
        raise e


# --- Standings & Results Processing Functions (for processor.py) ---
# Keep the more specific error handling in calculate_standings and fetch_race_results
# as they are run offline by processor.py, where specific errors are more useful.

def calculate_standings(year: int) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """ Calculates driver and team standings for a given year by processing race results. """
    # ... (Keep existing implementation with specific error handling) ...
    print(f"Calculating standings for {year}...")
    try:
        schedule = ff1.get_event_schedule(year, include_testing=False)
        completed_races = schedule[schedule['EventDate'].dt.tz_localize('UTC') < datetime.now(timezone.utc)]
        if completed_races.empty: return None, None
        driver_points = defaultdict(lambda: {'points': 0.0, 'wins': 0, 'podiums': 0, 'team': '', 'name': '', 'code': ''})
        team_points = defaultdict(lambda: {'points': 0.0, 'wins': 0, 'podiums': 0, 'team': '', 'color': 'gray'})
        print(f"Processing {len(completed_races)} completed events for {year}...")
        for index, event in completed_races.iterrows():
            print(f" Loading results for {event['EventName']}...")
            try:
                session = ff1.get_session(year, event['RoundNumber'], 'R')
                session.load(laps=False, telemetry=False, weather=False, messages=False) # Removed results=True
                if session.results is None or session.results.empty: continue
                for _, res in session.results.iterrows():
                    try:
                        driver_code = res.get('Abbreviation'); team_name = res.get('TeamName'); points = res.get('Points', 0.0)
                        position_val = res.get('Position'); status = str(res.get('Status', 'Unknown'))
                        last_name = res.get('LastName', 'N/A'); first_name = res.get('FirstName', '')
                        full_name = f"{first_name} {last_name}".strip()
                        driver_code = str(driver_code) if pd.notna(driver_code) else None
                        team_name = str(team_name) if pd.notna(team_name) else None
                        points = float(points) if pd.notna(points) else 0.0
                        try: position = int(float(position_val)) if pd.notna(position_val) and str(position_val).replace('.','',1).isdigit() else None
                        except (ValueError, TypeError): position = None
                        if not driver_code or not team_name: continue
                        driver_points[driver_code]['points'] += points; team_points[team_name]['points'] += points
                        driver_points[driver_code]['team'] = team_name; driver_points[driver_code]['name'] = full_name if full_name else last_name
                        driver_points[driver_code]['code'] = driver_code; team_points[team_name]['team'] = team_name
                        team_points[team_name]['color'] = get_team_color_name(team_name)
                        if status.startswith(('Finished', '+')) or status == 'Running':
                            if position == 1: driver_points[driver_code]['wins'] += 1; team_points[team_name]['wins'] += 1
                            if position is not None and position <= 3: driver_points[driver_code]['podiums'] += 1; team_points[team_name]['podiums'] += 1
                    except Exception as row_err: print(f"    !! Error processing row: {row_err}")
                del session; gc.collect(); time.sleep(0.1) # Shorter delay
            except Exception as race_err: print(f"  -> Error processing session {event['EventName']}: {race_err}") # Catch general exception here
        driver_df = pd.DataFrame(list(driver_points.values())) if driver_points else None
        teams_df = pd.DataFrame(list(team_points.values())) if team_points else None
        if driver_df is not None and not driver_df.empty:
            driver_df = driver_df.sort_values(by=['points', 'wins', 'podiums', 'code'], ascending=[False, False, False, True]).reset_index(drop=True)
            driver_df['rank'] = driver_df.index + 1
        if teams_df is not None and not teams_df.empty:
            teams_df = teams_df.sort_values(by=['points', 'wins', 'podiums', 'team'], ascending=[False, False, False, True]).reset_index(drop=True)
            teams_df['rank'] = teams_df.index + 1
            teams_df['shortName'] = teams_df['team'].apply(lambda x: x[:3].upper() if x else 'N/A')
        print(f"Finished calculating standings for {year}.")
        return driver_df, teams_df
    except Exception as e: print(f"Error calculating standings for {year}: {e}"); raise e


def fetch_race_results(year: int) -> list[dict] | None:
    """ Fetches basic race results (winner) for a given year. """
    print(f"Fetching race results summary for {year}...")
    try:
        schedule = ff1.get_event_schedule(year, include_testing=False)
        completed_races = schedule[schedule['EventDate'].dt.tz_localize('UTC') < datetime.now(timezone.utc)]
        if completed_races.empty: return []
        results_list = []
        print(f"Processing {len(completed_races)} completed events for {year}...")
        for index, event in completed_races.iterrows():
            try:
                session = ff1.get_session(year, event['RoundNumber'], 'R')
                session.load(laps=False, telemetry=False, weather=False, messages=False)
                if session.results is None or session.results.empty: continue
                winner = session.results[session.results['Position'] == 1].iloc[0] if not session.results[session.results['Position'] == 1].empty else None
                if winner is not None:
                     results_list.append({"year": year, "event": event['EventName'], "round": event['RoundNumber'],
                                          "driver": winner.get('LastName', 'N/A'), "team": winner.get('TeamName'),
                                          "teamColor": get_team_color_name(winner.get('TeamName'))})
                del session; gc.collect(); time.sleep(0.1)
            except Exception as race_err: print(f"  -> Error fetching winner for {event['EventName']}: {race_err}") # Catch general exception here
        results_list.sort(key=lambda x: x['round'])
        print(f"Finished fetching race results summary for {year}. Found {len(results_list)} results.")
        return results_list
    except Exception as e: print(f"Error fetching race results summary for {year}: {e}"); raise e

def fetch_and_process_steering(year: int, event: str, session_type: str, driver_code: str, lap_identifier: str) -> pd.DataFrame | None:
    """ Fetches telemetry for a specific lap and processes steering data. """
    print(f"Processing steering data for {driver_code} lap {lap_identifier} - {year} {event} {session_type}")
    try:
        session_to_load = map_session_identifier_for_load(session_type)
        print(f" -> Mapped session type {session_type} to {session_to_load} for loading")
        session = ff1.get_session(year, event, session_to_load)
        session.load(laps=True, telemetry=True, weather=False, messages=False)
        laps = session.laps.pick_drivers([driver_code])
        if laps.empty: return None
        target_lap = None
        if lap_identifier.lower() == 'fastest':
            target_lap = laps.pick_fastest()
            if target_lap is None or pd.isna(target_lap['LapTime']): target_lap = laps.iloc[-1] if not laps.empty else None
        else:
            try: lap_num = int(lap_identifier); lap_row = laps[laps['LapNumber'] == lap_num]; target_lap = lap_row.iloc[0] if not lap_row.empty else None
            except (ValueError, IndexError): return None
        if target_lap is None: return None
        telemetry = target_lap.get_car_data(pad=1)
        if telemetry.empty or 'Steer' not in telemetry.columns: return None
        if 'Distance' not in telemetry.columns: telemetry = telemetry.add_distance()
        steering_df = telemetry[['Distance', 'Steer']].copy().dropna(subset=['Distance', 'Steer']).replace({np.nan: None})
        # Rename 'Steer' to 'SteeringWheel' to match the expected column name in the frontend
        steering_df.rename(columns={'Steer': 'SteeringWheel'}, inplace=True)
        print(f"Successfully processed steering data for lap {getattr(target_lap, 'LapNumber', 'N/A')}. Records: {len(steering_df)}")
        return steering_df
    except Exception as e: print(f"Error processing steering data: {e}"); raise e

def fetch_and_process_throttle(year: int, event: str, session_type: str, driver_code: str, lap_identifier: str) -> pd.DataFrame | None:
    """ Fetches telemetry for a specific lap and processes throttle data. """
    print(f"Processing throttle data for {driver_code} lap {lap_identifier} - {year} {event} {session_type}")
    try:
        session_to_load = map_session_identifier_for_load(session_type)
        print(f" -> Mapped session type {session_type} to {session_to_load} for loading")
        session = ff1.get_session(year, event, session_to_load)
        session.load(laps=True, telemetry=True, weather=False, messages=False)
        laps = session.laps.pick_drivers([driver_code])
        if laps.empty: return None
        target_lap = None
        if lap_identifier.lower() == 'fastest':
            target_lap = laps.pick_fastest()
            if target_lap is None or pd.isna(target_lap['LapTime']): target_lap = laps.iloc[-1] if not laps.empty else None
        else:
            try: lap_num = int(lap_identifier); lap_row = laps[laps['LapNumber'] == lap_num]; target_lap = lap_row.iloc[0] if not lap_row.empty else None
            except (ValueError, IndexError): return None
        if target_lap is None: return None
        telemetry = target_lap.get_car_data(pad=1)
        if telemetry.empty or 'Throttle' not in telemetry.columns: return None
        if 'Distance' not in telemetry.columns: telemetry = telemetry.add_distance()
        throttle_df = telemetry[['Distance', 'Throttle']].copy().dropna(subset=['Distance', 'Throttle']).replace({np.nan: None})
        print(f"Successfully processed throttle data for lap {getattr(target_lap, 'LapNumber', 'N/A')}. Records: {len(throttle_df)}")
        return throttle_df
    except Exception as e: print(f"Error processing throttle data: {e}"); raise e

def fetch_and_process_brake(year: int, event: str, session_type: str, driver_code: str, lap_identifier: str) -> pd.DataFrame | None:
    """ Fetches telemetry for a specific lap and processes brake data. """
    print(f"Processing brake data for {driver_code} lap {lap_identifier} - {year} {event} {session_type}")
    try:
        session_to_load = map_session_identifier_for_load(session_type)
        print(f" -> Mapped session type {session_type} to {session_to_load} for loading")
        session = ff1.get_session(year, event, session_to_load)
        session.load(laps=True, telemetry=True, weather=False, messages=False)
        laps = session.laps.pick_drivers([driver_code])
        if laps.empty: return None
        target_lap = None
        if lap_identifier.lower() == 'fastest':
            target_lap = laps.pick_fastest()
            if target_lap is None or pd.isna(target_lap['LapTime']): target_lap = laps.iloc[-1] if not laps.empty else None
        else:
            try: lap_num = int(lap_identifier); lap_row = laps[laps['LapNumber'] == lap_num]; target_lap = lap_row.iloc[0] if not lap_row.empty else None
            except (ValueError, IndexError): return None
        if target_lap is None: return None
        telemetry = target_lap.get_car_data(pad=1)
        if telemetry.empty or 'Brake' not in telemetry.columns: return None
        if 'Distance' not in telemetry.columns: telemetry = telemetry.add_distance()
        
        # Get the Brake data
        brake_df = telemetry[['Distance', 'Brake']].copy().dropna(subset=['Distance', 'Brake']).replace({np.nan: None})
        
        # Convert boolean values to percentages (0 or 100)
        if brake_df['Brake'].dtype == bool:
            brake_df['Brake'] = brake_df['Brake'].apply(lambda x: 100 if x else 0)
        
        print(f"Successfully processed brake data for lap {getattr(target_lap, 'LapNumber', 'N/A')}. Records: {len(brake_df)}")
        return brake_df
    except Exception as e: print(f"Error processing brake data: {e}"); raise e

def fetch_and_process_rpm(year: int, event: str, session_type: str, driver_code: str, lap_identifier: str) -> pd.DataFrame | None:
    """ Fetches telemetry for a specific lap and processes RPM data. """
    print(f"Processing RPM data for {driver_code} lap {lap_identifier} - {year} {event} {session_type}")
    try:
        session_to_load = map_session_identifier_for_load(session_type)
        print(f" -> Mapped session type {session_type} to {session_to_load} for loading")
        session = ff1.get_session(year, event, session_to_load)
        session.load(laps=True, telemetry=True, weather=False, messages=False)
        laps = session.laps.pick_drivers([driver_code])
        if laps.empty: return None
        target_lap = None
        if lap_identifier.lower() == 'fastest':
            target_lap = laps.pick_fastest()
            if target_lap is None or pd.isna(target_lap['LapTime']): target_lap = laps.iloc[-1] if not laps.empty else None
        else:
            try: lap_num = int(lap_identifier); lap_row = laps[laps['LapNumber'] == lap_num]; target_lap = lap_row.iloc[0] if not lap_row.empty else None
            except (ValueError, IndexError): return None
        if target_lap is None: return None
        telemetry = target_lap.get_car_data(pad=1)
        if telemetry.empty or 'RPM' not in telemetry.columns: return None
        if 'Distance' not in telemetry.columns: telemetry = telemetry.add_distance()
        rpm_df = telemetry[['Distance', 'RPM']].copy().dropna(subset=['Distance', 'RPM']).replace({np.nan: None})
        print(f"Successfully processed RPM data for lap {getattr(target_lap, 'LapNumber', 'N/A')}. Records: {len(rpm_df)}")
        return rpm_df
    except Exception as e: print(f"Error processing RPM data: {e}"); raise e

def fetch_and_process_drs(year: int, event: str, session_type: str, driver_code: str, lap_identifier: str) -> pd.DataFrame | None:
    """ Fetches telemetry for a specific lap and processes DRS activation data. """
    print(f"Processing DRS data for {driver_code} lap {lap_identifier} - {year} {event} {session_type}")
    try:
        session_to_load = map_session_identifier_for_load(session_type)
        print(f" -> Mapped session type {session_type} to {session_to_load} for loading")
        session = ff1.get_session(year, event, session_to_load)
        session.load(laps=True, telemetry=True, weather=False, messages=False)

        # Use pick_drivers instead of pick_driver
        laps = session.laps.pick_drivers([driver_code])
        if laps.empty:
            print(f"No laps found for driver {driver_code}.")
            return None

        target_lap = None
        if lap_identifier.lower() == 'fastest':
            target_lap = laps.pick_fastest()
            # Fallback if fastest lap is incomplete or has no time
            if target_lap is None or pd.isna(target_lap['LapTime']):
                print(f"Fastest lap not usable for {driver_code}, falling back to last completed lap.")
                # Try finding the last lap with a recorded LapTime
                valid_laps = laps.dropna(subset=['LapTime'])
                target_lap = valid_laps.iloc[-1] if not valid_laps.empty else None
                if target_lap is None: # If still no valid lap, try the absolute last lap row
                     target_lap = laps.iloc[-1] if not laps.empty else None

        else:
            try:
                lap_num = int(lap_identifier)
                lap_row = laps[laps['LapNumber'] == lap_num]
                target_lap = lap_row.iloc[0] if not lap_row.empty else None
            except (ValueError, IndexError):
                print(f"Invalid lap identifier '{lap_identifier}' or lap not found.")
                return None

        if target_lap is None:
            print(f"Could not determine target lap for {driver_code}, identifier: {lap_identifier}.")
            return None

        lap_number_for_log = getattr(target_lap, 'LapNumber', 'N/A')
        print(f" -> Target lap identified: {lap_number_for_log}")

        telemetry = target_lap.get_car_data()
        if telemetry.empty or 'DRS' not in telemetry.columns:
            print(f"DRS telemetry data not found for lap {lap_number_for_log}.")
            return None

        if 'Distance' not in telemetry.columns:
            telemetry = telemetry.add_distance()

        # Filter for DRS activation points (where DRS value changes, focusing on activation 10, 12, 14)
        # Consider only points where DRS is active (values 10, 12, 14 usually mean enabled in zones)
        # We need Distance and the DRS status
        drs_data = telemetry[['Distance', 'DRS']].dropna()

        # Simplified: return points where DRS is likely active (>= 10 is a common indicator)
        drs_active_points = drs_data[drs_data['DRS'] >= 8] # Use 8 as threshold based on FastF1 examples

        if drs_active_points.empty:
             print(f"No DRS activation points found for lap {lap_number_for_log}.")
             return pd.DataFrame() # Return empty DataFrame

        # Convert to expected format if needed, here returning Distance and DRS value
        final_drs_df = drs_active_points[['Distance', 'DRS']].copy()
        final_drs_df['DRS_Active'] = 1 # Simple flag indicating DRS was active at this point

        print(f"Successfully processed DRS data for lap {lap_number_for_log}. Active points: {len(final_drs_df)}")
        # Return distance and a flag indicating DRS was active
        return final_drs_df[['Distance', 'DRS_Active']].replace({np.nan: None})

    except Exception as e:
        print(f"Error processing DRS data: {e}")
        import traceback
        traceback.print_exc()
        raise e

def fetch_session_incidents_and_results(year: int, event: str, session_type: str) -> tuple[list[dict], pd.DataFrame | None]:
    """
    Fetches incident messages for a session AND the official or provisional race results.
    If official results (session.results) are not available, it calculates
    provisional results based on lap timing data.
    Returns a tuple: (list_of_incidents, results_dataframe).
    The results DataFrame will have a 'Provisional' boolean column.
    """
    print(f"Fetching incidents and results for {year} {event} {session_type}")
    try:
        session_to_load = map_session_identifier_for_load(session_type)
        print(f" -> Mapped session type {session_type} to {session_to_load} for loading")
        session = ff1.get_session(year, event, session_to_load)

        # Load necessary data: laps are crucial for fallback, messages for incidents.
        # Results are loaded implicitly if available by fastf1 when accessing session.results
        session.load(laps=True, messages=True, weather=False, telemetry=False)
        print("Session data loaded (laps, messages).")

        # --- Get Results (Official or Provisional) ---
        results_df = session.results
        provisional = False

        # Check if official results are missing or empty
        if results_df is None or results_df.empty:
            print(f"WARNING: Official results for {year} {event} {session_type} not available via Ergast/session.results. Calculating provisional results from lap timing.")
            provisional = True
            laps = session.laps
            if laps is None or laps.empty:
                print("ERROR: No lap data available to calculate provisional results.")
                results_df = pd.DataFrame() # Return empty DataFrame if no laps
            else:
                # Find the last lap for each driver based on their latest timestamp
                # Ensure we handle potential NaT in Time before idxmax
                valid_time_laps = laps.dropna(subset=['Time'])
                if valid_time_laps.empty:
                     print("ERROR: No laps with valid timestamps found.")
                     results_df = pd.DataFrame()
                else:
                     last_laps_idx = valid_time_laps.groupby('Driver')['Time'].idxmax()
                     last_laps = laps.loc[last_laps_idx].copy()

                     # Get grid position if available from laps data
                     if 'GridPosition' in laps.columns:
                         # Get the earliest lap for each driver to reliably get starting grid pos
                         grid_positions = laps.loc[laps.groupby('Driver')['LapNumber'].idxmin()][['Driver', 'GridPosition']].dropna().drop_duplicates('Driver').set_index('Driver')
                         last_laps = last_laps.join(grid_positions, on='Driver')
                     else:
                         last_laps['GridPosition'] = 0 # Assign default if not found

                     # --- Data Enrichment (Team, FullName, DriverNumber) ---
                     # Attempt to get these from the session's drivers list if laps lack them
                     driver_info = {}
                     try:
                         # Access session.drivers which should be populated after load
                         # session.drivers is a dict {driver_code: DriverInfoObject}
                         session_drivers_info = getattr(session, 'drivers', None)
                         if session_drivers_info:
                            # Need driver number mapping if laps doesn't have it
                             for drv_num in session_drivers_info:
                                 info = session.get_driver(drv_num) # Get driver info object
                                 if info is not None:
                                     code = info.get('Abbreviation', f'UNKNOWN_{drv_num}') # Use Abbreviation as the key
                                     driver_info[code] = {
                                        'DriverNumber': info.get('DriverNumber', drv_num),
                                        'TeamName': info.get('TeamName', 'N/A'),
                                        'FullName': info.get('FullName', code)
                                     }
                         else: # Fallback: Try unique values from laps if session.drivers fails
                              if 'Team' in laps.columns:
                                  teams = laps[['Driver', 'Team']].dropna().drop_duplicates('Driver').set_index('Driver')['Team'].to_dict()
                              else: teams = {}
                              if 'FullName' in laps.columns:
                                  names = laps[['Driver', 'FullName']].dropna().drop_duplicates('Driver').set_index('Driver')['FullName'].to_dict()
                              else: names = {}
                              if 'DriverNumber' in laps.columns:
                                  numbers = laps[['Driver', 'DriverNumber']].dropna().drop_duplicates('Driver').set_index('Driver')['DriverNumber'].to_dict()
                              else: numbers = {}

                              for code in last_laps['Driver'].unique():
                                   driver_info[code] = {
                                       'DriverNumber': numbers.get(code, 'N/A'),
                                       'TeamName': teams.get(code, 'N/A'),
                                       'FullName': names.get(code, code) # Use code if no FullName
                                   }


                     except Exception as e:
                         print(f"Minor error getting driver details for enrichment: {e}")
                         # Continue without full enrichment if error occurs


                     # Apply enrichment to last_laps DataFrame
                     last_laps['TeamName'] = last_laps['Driver'].map(lambda x: driver_info.get(x, {}).get('TeamName', 'N/A'))
                     last_laps['FullName'] = last_laps['Driver'].map(lambda x: driver_info.get(x, {}).get('FullName', x))
                     last_laps['DriverNumber'] = last_laps['Driver'].map(lambda x: driver_info.get(x, {}).get('DriverNumber', 'N/A'))
                     # --- End Enrichment ---


                     # Sort by LapNumber descending, then Time ascending
                     last_laps = last_laps.sort_values(by=['LapNumber', 'Time'], ascending=[False, True])
                     last_laps = last_laps.reset_index(drop=True) # Use 0-based index for position

                     # Determine max laps for status calculation
                     max_laps_completed = last_laps['LapNumber'].max()

                     # Build the results DataFrame
                     provisional_results_list = []
                     for index, row in last_laps.iterrows():
                         status = 'Finished'
                         # Check if the driver completed the max laps
                         if row['LapNumber'] < max_laps_completed:
                             status = f"+{int(max_laps_completed - row['LapNumber'])} Lap(s)"
                         # TODO: Add better logic here if we can determine DNF/Retired status reliably from messages or lap patterns

                         # Use .get with defaults for safety, especially after enrichment attempts
                         provisional_results_list.append({
                             'DriverNumber': row.get('DriverNumber', 'N/A'),
                             'Abbreviation': row['Driver'], # 'Driver' column holds the abbreviation
                             'TeamName': row.get('TeamName', 'N/A'),
                             'FullName': row.get('FullName', row['Driver']),
                             'Position': index + 1, # Position based on final sort order
                             'GridPosition': int(row.get('GridPosition', 0)), # Ensure int
                             'Status': status,
                             'Time': row['Time'], # This is the finish time on their last lap (or last recorded time)
                             'Laps': int(row['LapNumber']), # Ensure int
                             # 'Points': 0 # Points are usually official, omit for provisional
                         })

                     results_df = pd.DataFrame(provisional_results_list)

                     # Define standard columns expected (similar to session.results)
                     standard_cols = ['DriverNumber', 'Abbreviation', 'FullName', 'TeamName', 'Position', 'GridPosition', 'Status', 'Time', 'Laps']
                     # Ensure all standard columns exist, fill missing with defaults if necessary
                     for col in standard_cols:
                         if col not in results_df.columns:
                             results_df[col] = None # Or appropriate default (e.g., 0 for Laps/Position?)

                     # Reorder and select standard columns
                     results_df = results_df[standard_cols]

        # Add the 'Provisional' flag column AFTER potentially creating the DataFrame
        if results_df is not None:
            # Ensure results_df is a DataFrame before adding column
             if not isinstance(results_df, pd.DataFrame):
                 print("Warning: results_df is not a DataFrame, cannot add Provisional column.")
                 # Handle case where official results were some other type? Unlikely.
             elif results_df.empty and provisional:
                 # If we intended to create provisional but failed (e.g., no laps),
                 # return an empty DF but maybe add the column still?
                 results_df['Provisional'] = provisional # Add to empty DF
             elif not results_df.empty:
                 results_df['Provisional'] = provisional # Add the flag to populated DF

        # --- Incident processing logic ---
        incidents = []
        messages = session.messages
        if messages is not None and not messages.empty:
             print(f"Processing {len(messages)} messages for incidents.")
             # Filter relevant messages (e.g., flags, SC, VSC, investigations)
             incident_keywords = ['FLAG', 'DEPLOYED', 'ENDED', 'CLEAR', 'INCIDENT', 'INVESTIGATED', 'PENALTY', 'STOP/GO', 'DRIVE THROUGH', 'RETIRED', 'PROBLEM', 'CRASH']
             # Simpler approach: Look for keywords in the 'Message' column (assuming it exists)
             # Note: session.messages structure might vary. Adapt based on actual columns.
             # Assuming a 'Message' or 'Category' column might exist. Check `session.messages.columns`
             if 'Message' in messages.columns:
                 potential_incidents = messages[messages['Message'].str.contains('|'.join(incident_keywords), case=False, na=False)].copy()
             elif 'Category' in messages.columns: # Alternative column name
                 potential_incidents = messages[messages['Category'].str.contains('|'.join(incident_keywords), case=False, na=False)].copy()
             else:
                  print("Warning: Cannot find 'Message' or 'Category' column in session.messages for incident detection.")
                  potential_incidents = pd.DataFrame() # Empty


             if not potential_incidents.empty:
                # Convert Time column to ISO format string if it's timedelta or datetime
                if 'Time' in potential_incidents.columns:
                    potential_incidents['TimeStr'] = potential_incidents['Time'].apply(lambda x: str(x)) # Simple string conversion
                else:
                    potential_incidents['TimeStr'] = 'N/A'

                # Use DriverNumber results mapping for abbreviations if available
                driver_abbr_map = {}
                if results_df is not None and not results_df.empty and 'DriverNumber' in results_df.columns and 'Abbreviation' in results_df.columns:
                     driver_abbr_map = results_df.set_index('DriverNumber')['Abbreviation'].to_dict()

                for _, row in potential_incidents.iterrows():
                    message_text = row.get('Message', row.get('Category', 'No Message Text')) # Get message content
                    # Try to map driver number if present in the message context
                    driver_number = row.get('RacingNumber', row.get('DriverNumber', None)) # Check common column names
                    driver_code = driver_abbr_map.get(driver_number, None)

                    incidents.append({
                        "time": row['TimeStr'],
                        "message": message_text,
                        "category": row.get('Category', 'Generic'), # Use category if exists
                        "flag": row.get('Flag', None), # Include flag if exists
                        "scope": row.get('Scope', None), # Include scope if exists
                        "driverCode": driver_code # Include driver code if mapped
                    })

        print(f"Found {len(incidents)} potential incidents.")
        # --- Return both incidents list and results DataFrame ---
        return incidents, results_df

    except ff1.ErgastUnavailableError:
         print("Ergast is unavailable. Cannot fetch official results or potentially some session info.")
         # Attempt to proceed with provisional results if possible (depends if get_session worked)
         # Re-running provisional logic here might be complex. Best to handle upstream or return error.
         # For now, return empty results / raise specific error?
         return [], pd.DataFrame({'Provisional': [True]}) # Return empty incidents and minimal provisional marker DF

    except ff1.FastF1Error as f1_error: # Catch specific FastF1 errors
        print(f"A FastF1 error occurred: {f1_error}")
        # Decide if we can still return provisional or need to raise
        # If session loading failed, we can't do much.
        raise f1_error # Re-raise for API handler
    except Exception as e:
        print(f"An unexpected error occurred in fetch_session_incidents_and_results: {e}")
        import traceback
        traceback.print_exc()
        # Return empty lists/frames to avoid breaking API structure if possible
        # Or re-raise depending on desired API robustness
        raise e

# Restore Stint Analysis processing function
def fetch_and_process_stint_analysis(year: int, event: str, session_type: str) -> list[dict] | None:
    """ Fetches lap data and processes detailed stint information including lap times. """
    print(f"Processing stint analysis - {year} {event} {session_type}")
    try:
        session_to_load = map_session_identifier_for_load(session_type)
        print(f" -> Mapped session type {session_type} to {session_to_load} for loading")
        session = ff1.get_session(year, event, session_to_load)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps
        if laps.empty:
            print(f"No lap data found for {year} {event} {session_type}.")
            return []

        drivers = laps['Driver'].unique()
        if len(drivers) == 0: return []

        stint_analysis_list = []
        for drv_code in drivers:
            try:
                # Use pick_drivers instead of pick_driver
                drv_laps = laps.pick_drivers([drv_code])
                if drv_laps.empty: continue

                # Ensure LapTime is converted to seconds
                if 'LapTimeSeconds' not in drv_laps.columns:
                    drv_laps['LapTimeSeconds'] = drv_laps['LapTime'].dt.total_seconds()

                # Check if 'TrackStatus' exists for filtering
                has_track_status = 'TrackStatus' in drv_laps.columns
                if not has_track_status:
                    print(f" - Warning: 'TrackStatus' column not found for driver {drv_code}. Cannot filter SC/VSC laps.")

                # Group by Stint
                stints_grouped = drv_laps.groupby("Stint")

                for stint_num, stint_laps_df in stints_grouped:
                    if stint_laps_df.empty: continue

                    # --- Filter for Green Flag Laps ---
                    if has_track_status:
                        # Assuming '1' represents green flag. Adjust if needed based on FastF1 version/data.
                        green_flag_laps_df = stint_laps_df[stint_laps_df['TrackStatus'] == '1'].copy()
                        if green_flag_laps_df.empty:
                            # print(f" - Driver {drv_code}, Stint {stint_num}: No green flag laps found in stint.")
                            # If no green laps, we'll skip this stint later based on empty lap_times_seconds
                            pass 
                    else:
                        # Fallback if TrackStatus column is missing
                        green_flag_laps_df = stint_laps_df.copy()
                    # ---------------------------------

                    # If filtering resulted in no green flag laps, skip this stint
                    if green_flag_laps_df.empty:
                        # print(f" - Driver {drv_code}, Stint {stint_num}: Skipping stint due to no green flag laps.")
                        continue

                    compound = green_flag_laps_df["Compound"].iloc[0]
                    # Use green flag laps for start/end, but this might be slightly confusing if first/last laps were SC
                    # Sticking with green flag laps for now as it reflects the laps used for timing.
                    start_lap = green_flag_laps_df["LapNumber"].min()
                    end_lap = green_flag_laps_df["LapNumber"].max()
                    
                    # Extract lap details (number and time in seconds) from the *filtered* DataFrame
                    lap_details = [
                        {"lapNumber": int(row["LapNumber"]), "lapTime": float(row["LapTimeSeconds"]) }
                        for _, row in green_flag_laps_df.iterrows() 
                        if pd.notna(row["LapTimeSeconds"])
                    ]

                    # Only add stint if there are valid lap times after filtering
                    if lap_details:
                        stint_analysis_list.append({
                            "driverCode": drv_code,
                            "stintNumber": int(stint_num),
                            "compound": compound,
                            "startLap": int(start_lap),
                            "endLap": int(end_lap),
                            "lapDetails": lap_details # List of {lapNumber, lapTime} dicts
                        })
                    # else:
                        # print(f" - Driver {drv_code}, Stint {stint_num}: Skipping stint as no valid lap times remain after dropna().")

            except Exception as inner_e:
                print(f"Error processing stint analysis for driver {drv_code}: {inner_e}")

        print(f"Successfully processed stint analysis for {len(drivers)} drivers.")
        return stint_analysis_list
    except Exception as e:
        print(f"Error processing stint analysis: {e}")
        import traceback
        traceback.print_exc()
        raise e

# --- Main Execution (for processor.py) ---
# (Keep __main__ block as is)
# ...
