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
import copy # Add copy for deep copying standings
# Removed specific FastF1 exception imports

# --- Configuration ---
# Get the directory where the script is located
script_dir = Path(__file__).resolve().parent

# Define cache paths relative to the script's directory
FASTF1_CACHE_PATH = os.getenv('FASTF1_CACHE_PATH', script_dir / 'cache')
DATA_CACHE_PATH = script_dir / "data_cache"

if not os.path.exists(FASTF1_CACHE_PATH):
    os.makedirs(FASTF1_CACHE_PATH)
ff1.Cache.enable_cache(FASTF1_CACHE_PATH)

# --- Helper Functions ---

def get_team_color_name(team_name: str | None) -> str:
    """Gets a simplified team color name."""
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
    if 'sauber' in simple_name: return 'alfaromeo' # Covers Kick Sauber too
    if 'racingbulls' in simple_name or 'alphatauri' in simple_name: return 'alphatauri'
    return 'gray'

def save_json(data, file_path: Path):
    """Saves data to a JSON file, handling potential numpy types."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            if isinstance(data, pd.DataFrame):
                # Convert DataFrame to list of dicts first
                records = data.replace({np.nan: None}).to_dict(orient='records')
                # Add pd.Timedelta handling to default serializer
                json.dump(records, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else str(x) if isinstance(x, pd.Timestamp) else x.total_seconds() if isinstance(x, pd.Timedelta) else None if pd.isna(x) else x)
            elif isinstance(data, (list, dict)):
                 # Handle potential numpy/pandas types within list/dict structures if necessary
                 # Add pd.Timedelta handling to default serializer
                 json.dump(data, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else str(x) if isinstance(x, pd.Timestamp) else x.total_seconds() if isinstance(x, pd.Timedelta) else None if pd.isna(x) else x)
            else:
                 print(f" -> Unsupported data type for JSON saving: {type(data)}")
                 return
        # print(f" -> Data successfully saved to {file_path}") # Reduce verbosity
    except Exception as e:
        print(f" -> Error saving JSON to {file_path}: {e}")
        import traceback
        traceback.print_exc()

def is_sprint_weekend(event_format: str) -> bool:
    """Check if event uses a sprint weekend format based on FastF1 EventFormat."""
    # Check if the format string contains 'sprint', case-insensitive
    return isinstance(event_format, str) and 'sprint' in event_format.lower()


def format_lap_time(lap_time_delta):
    """Formats a Pandas Timedelta lap time into MM:SS.ms string."""
    if pd.isna(lap_time_delta) or lap_time_delta is None:
        return None
    # Ensure it's a Timedelta object
    if not isinstance(lap_time_delta, pd.Timedelta):
        try:
            # Attempt conversion if it's a compatible type (like string)
            lap_time_delta = pd.to_timedelta(lap_time_delta)
        except (ValueError, TypeError):
            return None # Cannot convert

    total_seconds = lap_time_delta.total_seconds()
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int(round((total_seconds - int(total_seconds)) * 1000)) # Round milliseconds
    # Corrected format string for minutes
    return f"{minutes:d}:{seconds:02d}.{milliseconds:03d}"

def get_session_details(event):
    """Extract all sessions present in an event based on schedule data."""
    sessions = []
    # Check all possible session slots (Session1Type to Session5Type)
    for i in range(1, 6):
        session_type_key = f'Session{i}Type'
        session_date_key = f'Session{i}Date'
        # Safely get values using .get() on the Series (event)
        session_type = event.get(session_type_key)
        session_date = event.get(session_date_key)

        if pd.notna(session_type) and pd.notna(session_date):
            sessions.append({
                'type': session_type,
                'date': session_date # Keep as Timestamp for now
            })
    # Add Race session separately if EventDate exists
    race_date = event.get('EventDate')
    if pd.notna(race_date):
         sessions.append({
             'type': 'R',
             'date': race_date
         })
    # Sort sessions chronologically by date
    sessions.sort(key=lambda x: x['date'])
# Removed get_session_details function as logic is moved into process_season_data

def process_qualifying_segment(laps: pd.DataFrame, session_id: str) -> list[dict]:
    """Process qualifying segment results from laps data."""
    print(f"    -> Processing qualifying segment: {session_id}")
    if laps is None or laps.empty:
        print(f"    !! No lap data provided for segment {session_id}")
        return []

    # Filter for the specific segment (Q1, Q2, Q3, SQ1, etc.)
    # FastF1 uses 'Segment' column in recent versions. Need to check its existence.
    # This logic needs refinement based on actual FastF1 data structure for segments.
    # Assuming 'Segment' column exists for now.
    if 'Segment' not in laps.columns:
        print(f"    !! 'Segment' column not found in lap data for {session_id}. Cannot process segment.")
        return [] # Cannot process without segment info

    segment_laps = laps[laps['Segment'] == session_id].copy()
    if segment_laps.empty:
        print(f"    !! No laps found for segment {session_id}")
        return []

    accurate_laps = segment_laps.pick_accurate()
    processed = []

    # Get all drivers who participated in the segment
    all_segment_drivers_info = segment_laps[['Driver', 'Team', 'FullName']].drop_duplicates()

    if accurate_laps.empty:
        print(f"    !! No accurate laps found for segment {session_id}")
        # Return drivers who participated but set no time
        for _, driver_row in all_segment_drivers_info.iterrows():
             processed.append({
                 'position': None, 'driverCode': str(driver_row.get('Driver')),
                 'fullName': str(driver_row.get('FullName', 'N/A')),
                 'team': str(driver_row.get('Team', 'N/A')),
                 'teamColor': get_team_color_name(driver_row.get('Team')),
                 'status': 'No Time Set', 'fastestLapTime': None,
                 'lapsCompleted': len(segment_laps[segment_laps['Driver'] == driver_row.get('Driver')])
             })
        processed.sort(key=lambda x: x['fullName']) # Sort alphabetically if no times
        return processed

    # Get best lap for each driver in this segment
    best_laps = accurate_laps.loc[accurate_laps.groupby('Driver')['LapTime'].idxmin()]

    # Create results for drivers with accurate times
    drivers_with_times = set()
    for _, row in best_laps.iterrows():
        driver_code = str(row.get('Driver'))
        drivers_with_times.add(driver_code)
        processed.append({
            'position': None, # Will be assigned after sorting
            'driverCode': driver_code,
            'fullName': str(row.get('FullName', 'N/A')),
            'team': str(row.get('Team', 'N/A')),
            'teamColor': get_team_color_name(row.get('Team')),
            'status': 'Time Set',
            'fastestLapTime': format_lap_time(row.get('LapTime')),
            'lapsCompleted': len(segment_laps[segment_laps['Driver'] == driver_code])
        })

    # Add drivers who participated but didn't set an accurate time
    for _, driver_row in all_segment_drivers_info.iterrows():
        driver_code = str(driver_row.get('Driver'))
        if driver_code not in drivers_with_times:
             processed.append({
                 'position': None, 'driverCode': driver_code,
                 'fullName': str(driver_row.get('FullName', 'N/A')),
                 'team': str(driver_row.get('Team', 'N/A')),
                 'teamColor': get_team_color_name(driver_row.get('Team')),
                 'status': 'No Accurate Time', 'fastestLapTime': None,
                 'lapsCompleted': len(segment_laps[segment_laps['Driver'] == driver_code])
             })

    # Sort by lap time (None times go last), then assign position
    processed.sort(key=lambda x: (x['fastestLapTime'] is None, x['fastestLapTime']))
    for i, res in enumerate(processed):
        res['position'] = i + 1

    print(f"    -> Processed {len(processed)} drivers for segment {session_id}")
    return processed


def process_session_results(year: int, round_number: int, session_identifier: str):  # Changed to round_number
    """
    Process results for a specific session (FP1, Q1, R, Sprint, etc.)
    and save relevant columns to JSON.
    """
    print(f"  -> Processing results for: {year} Round {round_number} {session_identifier}")
    processed_results = []

    # Get event name from round number to maintain file naming
    try:
        schedule = ff1.get_event_schedule(year, include_testing=False) # Avoid loading schedule repeatedly if possible, but needed here for name
        event_details = schedule.get_event_by_round(round_number)
        if event_details is None or pd.isna(event_details.get('EventName')):
             print(f"    !! Could not find event name for year {year}, round {round_number}. Using generic name.")
             event_name = f"round_{round_number}"
        else:
             event_name = event_details['EventName']
    except Exception as e:
        print(f"    !! Error getting event name for round {round_number}: {e}. Using generic name.")
        event_name = f"round_{round_number}"

    event_slug = event_name.lower().replace(' ', '_')
    results_file = DATA_CACHE_PATH / str(year) / "races" / f"{event_slug}_{session_identifier}.json"

    try:
        # Determine parent session for segments
        parent_session_id = None
        is_segment = False
        if session_identifier in ['Q1', 'Q2', 'Q3']:
            parent_session_id = 'Q'
            is_segment = True
        elif session_identifier in ['SQ1', 'SQ2', 'SQ3']:
            parent_session_id = 'SQ'
            is_segment = True

        # Load session data using ROUND NUMBER
        session_to_load = parent_session_id if is_segment else session_identifier
        session_obj = ff1.get_session(year, round_number, session_to_load) # Critical fix here
        # Ensure laps are loaded for FP, Q, SQ, Segments, Race, and Sprint as needed for processing within this function
        load_laps = session_identifier.startswith(('FP', 'Q', 'SQ')) or is_segment or session_identifier in ['R', 'Sprint']
        # Load messages ONLY if loading Q or SQ parent session for segment processing later
        load_messages = session_to_load in ['Q', 'SQ']
        print(f"    -> Loading laps for {session_identifier}: {load_laps}, Loading messages: {load_messages}")
        session_obj.load(laps=load_laps, telemetry=False, weather=False, messages=load_messages) # Load messages conditionally

        results = session_obj.results
        laps_data = session_obj.laps if load_laps else None

        if results is None: results = pd.DataFrame() # Ensure results is DataFrame

        # --- Handle Qualifying/Sprint Qualifying Segments ---
        if is_segment:
            # Process segments using parent session.results, mapping segment ID to results column name
            if results is not None and not results.empty:
                segment_col_map = {'Q1': 'Q1', 'Q2': 'Q2', 'Q3': 'Q3',
                                   'SQ1': 'Q1', 'SQ2': 'Q2', 'SQ3': 'Q3'} # Ergast/FastF1 often use Q1/Q2/Q3 for SQ times in results
                segment_col_name = segment_col_map.get(session_identifier)

                if segment_col_name and segment_col_name in results.columns:
                    print(f"    -> Processing segment {session_identifier} using parent session results column '{segment_col_name}'...")
                    required_cols_segment = ['Abbreviation', 'TeamName', 'FullName', segment_col_name]
                    if not all(col in results.columns for col in required_cols_segment):
                         print(f"    !! Missing required columns in parent results for segment {session_identifier}. Columns: {results.columns.tolist()}")
                         processed_results = [] # Ensure empty list if columns missing
                    else:
                         segment_results = results[required_cols_segment].copy()
                         segment_results = segment_results.dropna(subset=[segment_col_name]) # Keep only drivers who set a time
                         segment_results = segment_results.sort_values(by=segment_col_name)
                         segment_results = segment_results.reset_index(drop=True)

                         for index, row in segment_results.iterrows():
                             driver_code = str(row.get('Abbreviation')) if pd.notna(row.get('Abbreviation')) else None
                             if not driver_code: continue
                             processed_results.append({
                                 'position': index + 1,
                                 'driverCode': driver_code,
                                 'fullName': str(row.get('FullName', 'N/A')),
                                 'team': str(row.get('TeamName', 'N/A')),
                                 'teamColor': get_team_color_name(row.get('TeamName')),
                                 'status': 'Time Set', # Assumed if they have a time for the segment
                                 'fastestLapTime': format_lap_time(row.get(segment_col_name)),
                                 'lapsCompleted': None # Laps completed per segment not easily available here
                             })
                         print(f"    -> Processed {len(processed_results)} drivers for segment {session_identifier} from results.")
                else:
                    print(f"    !! Column '{segment_col_name}' not found in parent session results for segment {session_identifier}. Cannot process segment from results.")
                    processed_results = [] # Ensure empty list if column missing
            else:
                print(f"    !! No parent session results found for {parent_session_id} to process segment {session_identifier}")

        # --- Handle FP, R, Sprint ---
        # This block now only handles non-segment sessions.
        elif not is_segment:
            if results.empty and session_identifier.startswith('FP') and laps_data is not None and not laps_data.empty:
                 print(f"    -> No structured results for {session_identifier}, using lap data...")
                 drivers_in_laps = laps_data[['Driver', 'Team', 'FullName']].drop_duplicates(subset=['Driver'])
                 for _, driver_row in drivers_in_laps.iterrows():
                     driver_code = str(driver_row.get('Driver'))
                     if not driver_code: continue
                     driver_laps = laps_data.pick_drivers([driver_code]) # Use pick_drivers
                     fastest = driver_laps.pick_fastest() if not driver_laps.empty else None
                     processed_results.append({
                         'position': None, 'driverCode': driver_code,
                         'fullName': str(driver_row.get('FullName', 'N/A')),
                         'team': str(driver_row.get('Team', 'N/A')),
                         'teamColor': get_team_color_name(driver_row.get('Team')),
                         'status': 'N/A', 'points': 0.0,
                         'fastestLapTime': format_lap_time(fastest['LapTime']) if fastest is not None else None,
                         'lapsCompleted': len(driver_laps) if not driver_laps.empty else 0
                     })
                 # Sort FP results by fastest lap time
                 processed_results.sort(key=lambda x: (x['fastestLapTime'] is None, x['fastestLapTime']))
                 for i, res in enumerate(processed_results):
                     res['position'] = i + 1

            elif not results.empty:
                # Process standard results DataFrame (R, Sprint, FP if results exist)
                print(f"    -> Processing standard results for {session_identifier}")

                # --- Pre-fetch fastest lap info for Race/Sprint ---
                fastest_lap_driver_code = None
                formatted_fastest_time = None
                if (session_identifier == 'R' or session_identifier == 'Sprint') and laps_data is not None and not laps_data.empty:
                    try:
                        fastest_lap = laps_data.pick_fastest()
                        if fastest_lap is not None and pd.notna(fastest_lap['Driver']) and pd.notna(fastest_lap['LapTime']):
                            fastest_lap_driver_code = str(fastest_lap['Driver'])
                            formatted_fastest_time = format_lap_time(fastest_lap['LapTime'])
                            print(f"      -> Fastest Lap: {fastest_lap_driver_code} ({formatted_fastest_time})")
                        else:
                            print("      -> Fastest lap data incomplete or not found.")
                    except Exception as fl_err:
                        print(f"      !! Error getting fastest lap: {fl_err}")
                # --- End pre-fetch ---


                for index, row in results.iterrows():
                    driver_code = str(row.get('Abbreviation')) if pd.notna(row.get('Abbreviation')) else None
                    if not driver_code: continue

                    result = {
                        'position': int(row['Position']) if pd.notna(row['Position']) else None,
                        'driverCode': driver_code,
                        'fullName': str(row.get('FullName', 'N/A')),
                        'team': str(row.get('TeamName', 'N/A')),
                        'teamColor': get_team_color_name(row.get('TeamName')),
                        'status': str(row.get('Status', 'N/A')),
                        # Initialize potentially missing fields
                        'isFastestLap': False,
                        'fastestLapTimeValue': None,
                        'poleLapTimeValue': None,
                    }

                    if session_identifier == 'R' or session_identifier == 'Sprint':
                        result['points'] = float(row.get('Points', 0.0)) if pd.notna(row.get('Points')) else 0.0
                        result['gridPosition'] = int(row.get('GridPosition')) if pd.notna(row.get('GridPosition')) else None
                        # Set fastest lap flag and time based on pre-fetched data
                        if driver_code == fastest_lap_driver_code:
                            result['isFastestLap'] = True
                            result['fastestLapTimeValue'] = formatted_fastest_time # Assign the fetched time

                        # Set pole lap time if grid position is 1 and Q3 time exists in results
                        if result['gridPosition'] == 1:
                             result['poleLapTimeValue'] = format_lap_time(row.get('Q3')) # Assumes Q3 time is in Race results

                    elif session_identifier.startswith('FP'):
                         result['points'] = 0.0
                         if laps_data is not None and not laps_data.empty:
                             driver_laps = laps_data.pick_drivers([driver_code]) # Use pick_drivers
                             fastest = driver_laps.pick_fastest() if not driver_laps.empty else None
                             result['fastestLapTime'] = format_lap_time(fastest['LapTime']) if fastest is not None else None
                             result['lapsCompleted'] = len(driver_laps) if not driver_laps.empty else 0
                         else:
                             result['fastestLapTime'] = None; result['lapsCompleted'] = 0
                    # Note: Q/SQ parent sessions are no longer processed in this block

                    processed_results.append(result)
                # Sort standard results by position (inside elif, after the loop)
                processed_results.sort(key=lambda x: (x['position'] is None, x['position']))
            else:
                # Use event_name which is defined earlier in the function
                print(f"    !! No results or lap data found for {year} {event_name} {session_identifier}")

        # Save the processed results (even if empty)
        save_json(processed_results, results_file)
        print(f"    -> Saved results ({len(processed_results)} drivers) for {session_identifier} to {results_file.name}")

        # --- Generate Chart Data (Moved Here) ---
        print(f"    -> Checking chart data generation for {session_identifier}...")
        charts_path = DATA_CACHE_PATH / str(year) / "charts" # Define charts path here
        charts_path.mkdir(parents=True, exist_ok=True)

        # --- Define Chart Filenames (Using UPPERCASE Session ID) --- #
        session_upper = session_identifier.upper()
        lap_times_file = charts_path / f"{event_slug}_{session_upper}_laptimes.json"
        tire_strategy_file = charts_path / f"{event_slug}_{session_upper}_tirestrategy.json"
        session_drivers_file = charts_path / f"{event_slug}_{session_upper}_drivers.json"
        positions_file = charts_path / f"{event_slug}_{session_upper}_positions.json"
        # speedtrace_file = charts_path / f"{event_slug}_{session_upper}_speedtrace.json" # Add if/when implemented
        # gearmap_file = charts_path / f"{event_slug}_{session_upper}_gearmap.json" # Add if/when implemented

        # Lap Times (FP, Q Segments, SQ Segments, Sprint, R)
        if laps_data is not None and not laps_data.empty:
            if not lap_times_file.exists():
                try:
                    print(f"      -> Generating lap times chart data for {session_identifier}...")
                    all_drivers_laps = laps_data['Driver'].unique()
                    laps_filtered = laps_data.pick_drivers(all_drivers_laps).pick_accurate().copy()
                    if not laps_filtered.empty:
                        laps_filtered.loc[:, 'LapTimeSeconds'] = laps_filtered['LapTime'].dt.total_seconds()
                        laps_pivot = laps_filtered.pivot_table(index='LapNumber', columns='Driver', values='LapTimeSeconds')
                        laps_pivot = laps_pivot.reset_index()
                        save_json(laps_pivot, lap_times_file)
                        print(f"        -> Saved lap times data to {lap_times_file.name}")
                    else:
                        print(f"        !! No accurate laps found for {session_identifier} lap times chart.")
                except Exception as chart_err:
                    print(f"      !! Error generating lap times chart for {session_identifier}: {chart_err}")
            else:
                print(f"      -> Lap times chart data already cached for {session_identifier}.")

            # Position Changes (Sprint, R)
            if session_identifier in ['Sprint', 'R']:
                if not positions_file.exists():
                    try:
                        print(f"      -> Generating position chart data for {session_identifier}...")
                        pos_data = laps_data[['LapNumber', 'Driver', 'Position']].dropna(subset=['Position'])
                        if not pos_data.empty:
                            pos_data['Position'] = pos_data['Position'].astype(int)
                            pos_pivot = pos_data.pivot_table(index='LapNumber', columns='Driver', values='Position')
                            pos_pivot = pos_pivot.reset_index()
                            save_json(pos_pivot, positions_file)
                            print(f"        -> Saved position data to {positions_file.name}")
                        else:
                            print(f"        !! No position data found in laps for {session_identifier} position chart.")
                    except Exception as chart_err:
                        print(f"      !! Error generating position chart for {session_identifier}: {chart_err}")
                else:
                    print(f"      -> Position chart data already cached for {session_identifier}.")

            # Tire Strategy (Sprint, R)
            if session_identifier in ['Sprint', 'R']:
                 if not tire_strategy_file.exists():
                    try:
                        print(f"      -> Generating tire strategy chart data for {session_identifier}...")
                        all_drivers_laps_strat = laps_data['Driver'].unique() # Use a different variable name just in case
                        strategy_list = []
                        for drv_code in all_drivers_laps_strat:
                            drv_laps = laps_data.pick_drivers([drv_code]) # Use pick_drivers
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
                        save_json(strategy_list, tire_strategy_file)
                        print(f"        -> Saved tire strategy data to {tire_strategy_file.name}")
                    except Exception as chart_err:
                        print(f"      !! Error generating tire strategy chart for {session_identifier}: {chart_err}")
                 else:
                    print(f"      -> Tire strategy chart data already cached for {session_identifier}.")
        else:
             print(f"    !! No lap data available for {session_identifier}, skipping lap-based chart generation.")

        # Session Drivers (All sessions) - Use processed_results generated earlier in this function
        if not session_drivers_file.exists():
            try:
                print(f"      -> Generating session drivers list for {session_identifier}...")
                if processed_results: # Check if processed_results is not empty
                    driver_list = [{"code": str(res['driverCode']), "name": res['fullName'], "team": res['team']}
                                   for res in processed_results if res.get('driverCode')]
                    driver_list.sort(key=lambda x: x['code'])
                    save_json(driver_list, session_drivers_file)
                    print(f"        -> Saved session drivers list to {session_drivers_file.name}")
                else:
                    print(f"        !! No processed results available for {session_identifier} drivers list.")
            except Exception as chart_err:
                print(f"      !! Error generating session drivers list for {session_identifier}: {chart_err}")
        else:
            print(f"      -> Session drivers list already cached for {session_identifier}.")
        # --- End Chart Data Generation ---

        return processed_results

    except Exception as e:
        print(f"    !! Error processing session results for {session_identifier}: {e}")
        # Attempt to save empty file on error
        try:
            if not results_file.exists():
                 save_json([], results_file)
                 print(f"    -> Saved empty results file for {session_identifier} due to error.")
        except Exception as save_err:
             print(f"    !! Failed to save empty results file after error: {save_err}")
        return None


def _process_session_points(results: pd.DataFrame | None,
                           driver_standings: defaultdict,
                           team_standings: defaultdict,
                           is_sprint_session: bool):
    """Universal points processing for any session type (Sprint or Race)."""
    if results is None or results.empty or not isinstance(results, pd.DataFrame):
        print(f"    !! No results data to process for points (Sprint={is_sprint_session}).")
        return

    print(f"    -> Processing points from {'Sprint' if is_sprint_session else 'Race'} session...")
    points_added_driver = 0
    points_added_team = 0
    # Ensure required columns exist
    required_cols = ['Abbreviation', 'TeamName', 'Points', 'FullName']
    if not all(col in results.columns for col in required_cols):
        print(f"    !! Missing required columns for points processing in {'Sprint' if is_sprint_session else 'Race'} results: { [col for col in required_cols if col not in results.columns] }")
        return

    for _, res in results.iterrows():
        try:
            driver_code = str(res.get('Abbreviation')) if pd.notna(res.get('Abbreviation')) else None
            team_name = str(res.get('TeamName')) if pd.notna(res.get('TeamName')) else None
            points = float(res.get('Points', 0.0)) if pd.notna(res.get('Points')) else 0.0

            if not driver_code or not team_name: continue

            # Initialize standings dicts if driver/team is new
            driver_standings[driver_code].setdefault('points', 0.0)
            driver_standings[driver_code].setdefault('sprint_points', 0.0)
            driver_standings[driver_code].setdefault('wins', 0)
            driver_standings[driver_code].setdefault('podiums', 0)
            team_standings[team_name].setdefault('points', 0.0)
            team_standings[team_name].setdefault('sprint_points', 0.0)
            team_standings[team_name].setdefault('wins', 0)
            team_standings[team_name].setdefault('podiums', 0)

            # Update total points
            driver_standings[driver_code]['points'] += points
            team_standings[team_name]['points'] += points
            points_added_driver += points
            points_added_team += points

            # Update driver/team info (can be updated by either sprint or race)
            # Ensure 'FullName' exists before accessing
            full_name = res.get('FullName', 'N/A')
            driver_standings[driver_code]['team'] = team_name
            driver_standings[driver_code]['name'] = full_name if pd.notna(full_name) else 'N/A'
            driver_standings[driver_code]['code'] = driver_code
            team_standings[team_name]['team'] = team_name
            team_color = get_team_color_name(team_name) # Get color once
            team_standings[team_name]['color'] = team_color

            # Track sprint points separately
            if is_sprint_session:
                driver_standings[driver_code]['sprint_points'] += points
                team_standings[team_name]['sprint_points'] += points

            # --- IMPORTANT: Win/Podium counting moved to after Race processing ---

        except KeyError as ke: print(f"    !! Missing key processing session points row: {ke}")
        except Exception as row_err: print(f"    !! Error processing session points row: {row_err}")
    print(f"    -> Added {points_added_driver:.1f} driver points and {points_added_team:.1f} team points from {'Sprint' if is_sprint_session else 'Race'}.")


# --- Main Data Processing Function ---

def process_season_data(year: int):
    """
    Processes all completed race weekends (including Sprints) for a given year,
    calculates standings, and saves results, standings, and chart data to JSON files.
    """
    print(f"--- Starting data processing for {year} ---")
    year_cache_path = DATA_CACHE_PATH / str(year)
    year_cache_path.mkdir(parents=True, exist_ok=True)
    races_path = year_cache_path / "races"
    races_path.mkdir(parents=True, exist_ok=True)
    standings_path = year_cache_path / "standings"
    standings_path.mkdir(parents=True, exist_ok=True)
    charts_path = year_cache_path / "charts"
    charts_path.mkdir(parents=True, exist_ok=True)

    try:
        schedule = ff1.get_event_schedule(year, include_testing=False)
        # Ensure EventDate is timezone-aware for comparison
        if schedule['EventDate'].dt.tz is None:
             schedule['EventDate'] = schedule['EventDate'].dt.tz_localize('UTC')
        completed_events = schedule[schedule['EventDate'] < datetime.now(timezone.utc)]


        if completed_events.empty:
            print(f"No completed events found for {year} yet.")
            if not (standings_path / "standings.json").exists():
                 save_json({"drivers": [], "teams": []}, standings_path / "standings.json")
            return

        print(f"Found {len(completed_events)} completed events for {year}.")
        is_ongoing = len(completed_events) < len(schedule)
        print(f"Season ongoing: {is_ongoing}")

        # Cumulative standings dictionaries
        driver_standings = defaultdict(lambda: {'points': 0.0, 'sprint_points': 0.0, 'wins': 0, 'podiums': 0, 'team': '', 'name': '', 'code': ''})
        team_standings = defaultdict(lambda: {'points': 0.0, 'sprint_points': 0.0, 'wins': 0, 'podiums': 0, 'team': '', 'color': 'gray'})
        previous_driver_standings = {}
        previous_team_standings = {}
        all_race_results_summary = []

        num_completed = len(completed_events)
        for i, (index, event) in enumerate(completed_events.iterrows()):
            event_name = event['EventName']
            round_number = event['RoundNumber']
            event_format = event['EventFormat']
            # Ensure event_slug is lowercase for consistent file naming
            event_slug = event_name.lower().replace(' ', '_')

            print(f"\nProcessing event: {event_name} (Round {round_number}, Format: {event_format})...")
            # Chart file paths are now defined within process_session_results

            # Snapshot standings before processing the *last* completed event if season is ongoing
            if is_ongoing and i == num_completed - 1:
                print(f"  -> Snapshotting standings before processing final completed event: {event_name}")
                previous_driver_standings = copy.deepcopy(driver_standings)
                previous_team_standings = copy.deepcopy(team_standings)

            # --- Determine and Process All Sessions/Segments using Schedule Row ---
            # Removed debug print
            sessions_to_process = []
            q_session_laps = None # Store loaded Q laps to avoid reloading
            sq_session_laps = None # Store loaded SQ laps

            try:
                # Use the 'event' Series from the loop directly
                # Map FastF1 session names to our identifiers
                session_mapping = {
                    'Practice 1': 'FP1', 'Practice 2': 'FP2', 'Practice 3': 'FP3',
                    'Qualifying': 'Q', 'Sprint Qualifying': 'SQ',
                    'Sprint': 'Sprint', 'Race': 'R'
                }

                initial_session_identifiers = []
                # Iterate through Session<N> columns in the event Series (Corrected Key)
                for i in range(1, 6):
                    session_type_key = f'Session{i}' # Use Session<N> as the key for the session name
                    session_date_key = f'Session{i}Date'
                    # Safely get values using .get() on the Series (event)
                    session_type = event.get(session_type_key) # This now gets the session name (e.g., 'Practice 1')
                    session_date = event.get(session_date_key)

                    if pd.notna(session_type) and pd.notna(session_date):
                        mapped_id = session_mapping.get(session_type)
                        if mapped_id:
                            initial_session_identifiers.append(mapped_id)
                        else:
                            print(f"    !! Unmapped session type from schedule: {session_type}")

                # Add Race ('R') identifier separately if EventDate exists
                race_date = event.get('EventDate')
                if pd.notna(race_date):
                    if 'R' not in initial_session_identifiers:
                        initial_session_identifiers.append('R')

                print(f"  -> Initial mapped identifiers from schedule row: {initial_session_identifiers}")

                # Expand Q/SQ into segments based on actual data
                processed_parents = set() # Keep track of Q/SQ parents processed into segments
                for identifier in list(initial_session_identifiers): # Iterate over a copy
                    if identifier == 'Q':
                        added_segments = False
                        try:
                            qual_session = ff1.get_session(year, round_number, 'Q')
                            qual_session.load(laps=True, telemetry=False, weather=False, messages=False)
                            q_session_laps = qual_session.laps # Store laps
                            has_q1, has_q2, has_q3 = False, False, False
                            if q_session_laps is not None and not q_session_laps.empty:
                                # Check based on results columns first (more reliable if present)
                                if qual_session.results is not None and not qual_session.results.empty:
                                    has_q1 = 'Q1' in qual_session.results.columns and pd.notna(qual_session.results['Q1']).any()
                                    has_q2 = 'Q2' in qual_session.results.columns and pd.notna(qual_session.results['Q2']).any()
                                    has_q3 = 'Q3' in qual_session.results.columns and pd.notna(qual_session.results['Q3']).any()
                                # Fallback to laps if results columns are missing/empty
                                else:
                                    print("    -> Qualifying results columns missing, checking laps for segments.")
                                    if 'Segment' in q_session_laps.columns:
                                        segments_present = q_session_laps['Segment'].unique()
                                        has_q1 = 'Q1' in segments_present
                                        has_q2 = 'Q2' in segments_present
                                        has_q3 = 'Q3' in segments_present
                                    else: has_q1 = True # Assume Q1 always happens if Q exists

                            if has_q3:
                                sessions_to_process.extend(['Q1', 'Q2', 'Q3']); added_segments = True
                            elif has_q2:
                                sessions_to_process.extend(['Q1', 'Q2']); added_segments = True
                            elif has_q1:
                                sessions_to_process.append('Q1'); added_segments = True

                            if added_segments:
                                processed_parents.add('Q') # Mark Q as processed into segments
                            else:
                                sessions_to_process.append('Q') # Fallback if no segments detected

                            del qual_session # Clean up session object
                        except Exception as q_err:
                            print(f"    !! Error checking qualifying segments, falling back to 'Q': {q_err}")
                            if 'Q' not in sessions_to_process: sessions_to_process.append('Q') # Ensure Q is added on error if not already
                            sessions_to_process.append('Q')
                    elif identifier == 'SQ':
                        added_segments = False
                        try:
                            sq_session = ff1.get_session(year, round_number, 'SQ')
                            sq_session.load(laps=True, telemetry=False, weather=False, messages=False)
                            sq_session_laps = sq_session.laps # Store laps
                            has_sq1, has_sq2, has_sq3 = False, False, False
                            if sq_session_laps is not None and not sq_session_laps.empty:
                                # Check results first
                                if sq_session.results is not None and not sq_session.results.empty:
                                    has_sq1 = 'Q1' in sq_session.results.columns and pd.notna(sq_session.results['Q1']).any() # Ergast uses Q1/Q2/Q3 for SQ times
                                    has_sq2 = 'Q2' in sq_session.results.columns and pd.notna(sq_session.results['Q2']).any()
                                    has_sq3 = 'Q3' in sq_session.results.columns and pd.notna(sq_session.results['Q3']).any()
                                # Fallback to laps
                                else:
                                    print("    -> Sprint Qualifying results columns missing, checking laps for segments.")
                                    if 'Segment' in sq_session_laps.columns:
                                        segments_present = sq_session_laps['Segment'].unique()
                                        has_sq1 = 'SQ1' in segments_present # FastF1 laps might use SQ1/SQ2/SQ3
                                        has_sq2 = 'SQ2' in segments_present
                                        has_sq3 = 'SQ3' in segments_present
                                    else: has_sq1 = True # Assume SQ1 always happens if SQ exists

                            if has_sq3:
                                sessions_to_process.extend(['SQ1', 'SQ2', 'SQ3']); added_segments = True
                            elif has_sq2:
                                sessions_to_process.extend(['SQ1', 'SQ2']); added_segments = True
                            elif has_sq1:
                                sessions_to_process.append('SQ1'); added_segments = True

                            if added_segments:
                                processed_parents.add('SQ') # Mark SQ as processed into segments
                            else:
                                sessions_to_process.append('SQ') # Fallback if no segments detected

                            del sq_session
                        except Exception as sq_err:
                            print(f"    !! Error checking sprint qualifying segments, falling back to 'SQ': {sq_err}")
                            if 'SQ' not in sessions_to_process: sessions_to_process.append('SQ') # Ensure SQ is added on error if not already
                            sessions_to_process.append('SQ')
                    # Only add non-parent identifiers if they haven't been processed into segments
                    elif identifier not in processed_parents:
                        sessions_to_process.append(identifier)

            except Exception as event_err:
                 print(f"  !! Error during session identification for {year} Round {round_number}: {event_err}")
                 # Fallback to trying just Race if event loading fails
                 sessions_to_process = ['R']


            # Remove duplicates and maintain rough order
            session_order = ['FP1','FP2','FP3','SQ1','SQ2','SQ3','Sprint','Q1','Q2','Q3','Q','R'] # Define preferred order
            final_session_list = sorted(list(set(sessions_to_process)), key=lambda x: session_order.index(x) if x in session_order else 99)
            print(f"  -> Sessions/Segments to process: {final_session_list}")

            # Process each identified session/segment (FP, Q segments, SQ segments)
            # Points accumulation happens separately below for Sprint and Race
            # race_session_results_for_charts is no longer needed here
            for session_id in final_session_list:
                if session_id not in ['Sprint', 'R']: # Process non-points sessions first
                    # Pass loaded laps to avoid reloading if processing segments
                    laps_to_pass = None
                    if session_id in ['Q1', 'Q2', 'Q3']:
                        laps_to_pass = q_session_laps
                    elif session_id in ['SQ1', 'SQ2', 'SQ3']:
                        laps_to_pass = sq_session_laps
                    # Pass ROUND NUMBER instead of event name
                    # No longer need laps_to_pass as segments are handled differently
                    process_session_results(year, round_number, session_id)

            # --- Process Sprint Session (if applicable) ---
            is_sprint = is_sprint_weekend(event_format)
            sprint_session_results = None
            if is_sprint:
                print(f"  -> Processing Sprint session for {event_name}...")
                try:
                    sprint_session = ff1.get_session(year, round_number, 'Sprint')
                    sprint_session.load(laps=False, telemetry=False, weather=False, messages=False)
                    sprint_session_results = sprint_session.results # Store for potential use
                    if sprint_session_results is not None and not sprint_session_results.empty:
                        # Process points from Sprint
                        _process_session_points(sprint_session_results, driver_standings, team_standings, is_sprint_session=True)
                        # Save Sprint results (optional, if needed separately)
                        sprint_results_file = races_path / f"{event_slug}_Sprint.json"
                        save_json(sprint_session_results, sprint_results_file)
                        print(f"    -> Saved Sprint results to {sprint_results_file.name}")
                    else:
                        print(f"    !! No results found for Sprint session {event_name}")
                    del sprint_session
                except Exception as sprint_err:
                    print(f"    !! Error processing sprint session for {event_name}: {sprint_err}")

            # --- Process Race Session ---
            print(f"  -> Processing Race session for {event_name}...")
            race_session_results = None # Define before try block
            try:
                race_session = ff1.get_session(year, round_number, 'R')
                # Load laps only if chart data needs generating
                # Force loading laps for charts to ensure data is always processed
                load_laps_for_charts = True # Force load laps
                # load_laps_for_charts = not all([
                #     lap_times_file.exists(), tire_strategy_file.exists(),
                #     session_drivers_file.exists(), positions_file.exists()
                # ])
                print(f"    -> Loading laps for Race charts: {load_laps_for_charts}")
                race_session.load(laps=load_laps_for_charts, telemetry=False, weather=False, messages=False)
                race_session_results = race_session.results # Store for points and charts

                if race_session_results is not None and not race_session_results.empty:
                    # Process points from Race
                    _process_session_points(race_session_results, driver_standings, team_standings, is_sprint_session=False)

                    # --- Count Wins and Podiums (ONLY from Race results) ---
                    print("    -> Counting Wins and Podiums from Race results...")
                    for _, res in race_session_results.iterrows():
                        driver_code = str(res.get('Abbreviation')) if pd.notna(res.get('Abbreviation')) else None
                        team_name = str(res.get('TeamName')) if pd.notna(res.get('TeamName')) else None
                        position = res.get('Position')

                        if not driver_code or not team_name or pd.isna(position): continue
                        position = int(position)

                        if position == 1:
                            driver_standings[driver_code]['wins'] += 1
                            team_standings[team_name]['wins'] += 1
                        if position <= 3:
                            driver_standings[driver_code]['podiums'] += 1
                            team_standings[team_name]['podiums'] += 1

                    # Save Race results (processed format) by calling process_session_results
                    # This now also handles chart generation internally
                    process_session_results(year, round_number, 'R')

                    # --- Chart Data processing is now handled within process_session_results ---

                else: # Handle case where race results might be empty
                    print(f"    !! No results found for Race session {event_name}")
                    # Still attempt to save empty file if needed
                    race_results_file = races_path / f"{event_slug}_R.json"
                    if not race_results_file.exists():
                         save_json([], race_results_file)
                         print(f"    -> Saved empty Race results file as no results were found.")


                # Clean up session object if it was loaded
                if 'race_session' in locals() and race_session is not None:
                    del race_session
            except Exception as race_err:
                    print(f"    !! Error processing race session for {event_name}: {race_err}")
                    # Attempt to save empty file if race results processing failed entirely
                    race_results_file = races_path / f"{event_slug}_R.json"
                    if not race_results_file.exists():
                        try:
                            save_json([], race_results_file)
                            print(f"    -> Saved empty Race results file due to error.")
                        except Exception as save_err:
                            print(f"    !! Failed to save empty Race results file after error: {save_err}")

            # Add winner to summary list (only from Race session results)
            # Need to load the race results file now as race_session_results_for_charts is gone
            race_results_file = races_path / f"{event_slug}_R.json"
            if race_results_file.exists():
                try:
                    with open(race_results_file, 'r') as f:
                        race_results_data = json.load(f)
                    if race_results_data: # Check if list is not empty
                         winner = next((res for res in race_results_data if res.get('position') == 1), None)
                         if winner:
                             all_race_results_summary.append({
                                 "year": year, "event": event_name, "round": round_number,
                                 "driver": winner.get('fullName', 'N/A'),
                                 "team": winner.get('team', 'N/A'),
                                 "teamColor": winner.get('teamColor', 'gray')
                             })
                except Exception as read_err:
                     print(f"    !! Error reading race results file {race_results_file.name} for winner summary: {read_err}")
            else:
                 print(f"    !! Race results file {race_results_file.name} not found for winner summary.")

            # --- Cleanup ---
            # Removed duplicated gc.collect() and time.sleep() here
            gc.collect()
            time.sleep(0.2) # Slightly shorter sleep

        # --- Final Standings Calculation & Saving ---
        print("\n--- Calculating Final Standings ---")
        driver_standings_list = []
        for code, driver in driver_standings.items():
            race_points = driver['points'] - driver.get('sprint_points', 0.0)
            driver_data = {
                'code': code,
                'name': driver['name'],
                'team': driver['team'],
                'total_points': driver['points'],
                'race_points': race_points,
                'sprint_points': driver.get('sprint_points', 0.0),
                'wins': driver['wins'],
                'podiums': driver['podiums'],
                'teamColor': get_team_color_name(driver['team'])
            }
            driver_standings_list.append(driver_data)

        team_standings_list = []
        for name, team in team_standings.items():
            race_points = team['points'] - team.get('sprint_points', 0.0)
            team_data = {
                'team': name,
                'total_points': team['points'],
                'race_points': race_points,
                'sprint_points': team.get('sprint_points', 0.0),
                'wins': team['wins'],
                'podiums': team['podiums'],
                'color': team['color'],
                'shortName': name[:3].upper() if name else 'N/A'
            }
            team_standings_list.append(team_data)

        # Calculate points change if season is ongoing
        if is_ongoing:
            print("Calculating points change for ongoing season...")
            for driver_data in driver_standings_list:
                prev_total_points = previous_driver_standings.get(driver_data['code'], {}).get('points', 0.0)
                driver_data['points_change'] = driver_data['total_points'] - prev_total_points
            for team_data in team_standings_list:
                prev_total_points = previous_team_standings.get(team_data['team'], {}).get('points', 0.0)
                team_data['points_change'] = team_data['total_points'] - prev_total_points

        # Sort standings
        driver_standings_list.sort(key=lambda x: (x['total_points'], x['wins'], x['podiums'], x['race_points']), reverse=True)
        for i, driver in enumerate(driver_standings_list):
            driver['rank'] = i + 1

        team_standings_list.sort(key=lambda x: (x['total_points'], x['wins'], x['podiums'], x['race_points']), reverse=True)
        for i, team in enumerate(team_standings_list):
            team['rank'] = i + 1

        # Prepare final standings structure for JSON
        final_driver_standings = [
            {
                'rank': d['rank'], 'code': d['code'], 'name': d['name'], 'team': d['team'],
                'points': d['total_points'], 'wins': d['wins'], 'podiums': d['podiums'],
                'points_change': d.get('points_change'), 'teamColor': d['teamColor']
            } for d in driver_standings_list
        ]
        final_team_standings = [
            {
                'rank': t['rank'], 'team': t['team'], 'points': t['total_points'],
                'wins': t['wins'], 'podiums': t['podiums'],
                'points_change': t.get('points_change'), 'teamColor': t['color'],
                'shortName': t['shortName']
            } for t in team_standings_list
        ]
        final_standings = {"drivers": final_driver_standings, "teams": final_team_standings}

        # Save final standings
        save_json(final_standings, standings_path / "standings.json")
        print(f"Saved final standings for {year}.")

        # Save race results summary (only main races)
        all_race_results_summary.sort(key=lambda x: x['round'])
        save_json(all_race_results_summary, year_cache_path / "race_results.json")
        print(f"Saved race results summary for {year}.")

    except Exception as e:
        print(f"An critical error occurred during {year} processing: {e}")
        import traceback
        traceback.print_exc()


# --- Main Execution ---
if __name__ == "__main__":
    target_years = [2025] # Add more years if needed
    current_year = datetime.now().year
    if current_year not in target_years: target_years.append(current_year)

    # Process years in reverse order (most recent first)
    for year in sorted(list(set(target_years)), reverse=False):
        process_season_data(year) # This function has internal error handling per year

    print("--- Data processing complete ---") # Ensure this is the final message
