import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
import folium
import time


# -------------------------
# 1) DATABASE INITIALIZATION
# -------------------------
conn = sqlite3.connect('uber_simulation.db')
c = conn.cursor()

# Create tables for cities, prime locations, drivers, ride requests
c.executescript("""
DROP TABLE IF EXISTS Cities;
DROP TABLE IF EXISTS PrimeLocations;
DROP TABLE IF EXISTS Drivers;
DROP TABLE IF EXISTS RideRequests;

CREATE TABLE Cities (
    city_id INTEGER PRIMARY KEY AUTOINCREMENT,
    city_name TEXT
);

CREATE TABLE PrimeLocations (
    prime_id INTEGER PRIMARY KEY AUTOINCREMENT,
    city_id INTEGER,
    location_name TEXT,
    latitude REAL,
    longitude REAL
);

CREATE TABLE Drivers (
    driver_id INTEGER PRIMARY KEY AUTOINCREMENT,
    driver_name TEXT,
    city TEXT,
    base_prime_id INTEGER,
    shift_start INTEGER,
    shift_end INTEGER,
    current_lat REAL,
    current_lon REAL,
    status TEXT,
    last_update_time TEXT
);

CREATE TABLE RideRequests (
    request_id INTEGER PRIMARY KEY AUTOINCREMENT,
    driver_id INTEGER,
    city TEXT,
    sim_date TEXT,
    requested_time TEXT,
    source_lat REAL,
    source_lon REAL,
    dest_lat REAL,
    dest_lon REAL,
    est_time INTEGER,
    fare REAL,
    cost REAL,
    status TEXT
);
""")
conn.commit()


# -------------------------
# 2) INITIALIZE DATA
# -------------------------
cities = ["Delhi"]
prime_locations = [
    {"city": "Delhi", "name": f"Prime{i+1}", "lat": 28.6+random.uniform(-0.02,0.02), "lon":77.2+random.uniform(-0.02,0.02)}
    for i in range(5)
]

# Insert cities
for city in cities:
    c.execute("INSERT INTO Cities (city_name) VALUES (?)", (city,))
conn.commit()

# Insert prime locations
for loc in prime_locations:
    city_id = 1  # since only one city now
    c.execute("""
        INSERT INTO PrimeLocations (city_id, location_name, latitude, longitude)
        VALUES (?, ?, ?, ?)
    """, (city_id, loc["name"], loc["lat"], loc["lon"]))
conn.commit()

# Insert drivers (50 drivers for 5 primes)
drivers = []
for prime_id in range(1,6):
    for i in range(10):  # 10 drivers per prime
        driver_name = f"Driver_P{prime_id}_{i+1}"
        base_lat, base_lon = prime_locations[prime_id-1]["lat"], prime_locations[prime_id-1]["lon"]
        drivers.append((driver_name, "Delhi", prime_id, -1, -1, base_lat, base_lon, "idle", datetime.now().isoformat()))
c.executemany("""
    INSERT INTO Drivers (driver_name, city, base_prime_id, shift_start, shift_end, current_lat, current_lon, status, last_update_time)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
""", drivers)


# Assign driver shifts after inserting fresh drivers
drivers_df = pd.read_sql("SELECT driver_id FROM Drivers ORDER BY driver_id", conn)
num_drivers = len(drivers_df)

# Build group sizes as before
group_sizes = [num_drivers // 4] * 4
for i in range(num_drivers % 4):
    group_sizes[i] += 1

time_windows = [
    (5, 15),   # 5am - 3pm
    (9, 19),   # 9am - 7pm
    (13, 23),  # 1pm - 11pm
    (19, 29),  # 7pm - 5am next day (wraps past midnight)
]

driver_iter = iter(drivers_df.itertuples(index=False))
for group_num, group_size in enumerate(group_sizes):
    shift_start, shift_end = time_windows[group_num]
    for _ in range(group_size):
        try:
            driver = next(driver_iter)
        except StopIteration:
            raise ValueError("🚨 Ran out of drivers during shift assignment — check driver count vs. group_sizes!")
        driver_id = driver.driver_id
        c.execute(
            "UPDATE Drivers SET shift_start=?, shift_end=? WHERE driver_id=?",
            (shift_start, shift_end, driver_id)
        )

conn.commit()

# Debug: check assignments
#drivers_df_debug = pd.read_sql("SELECT driver_id, shift_start, shift_end FROM Drivers ORDER BY driver_id", conn)
#print("✅ Assigned shifts:\n", drivers_df_debug)


'''
# -------------------------
# 3) RIDE REQUEST SIMULATION
# -------------------------
target_completed_rides = 500
today = datetime.now().strftime('%A')

# Weekday multiplier
multiplier = 1.2 if today in ['Monday','Friday','Saturday'] else 1.0
num_successful = int(target_completed_rides * multiplier)
num_requests = num_successful * 3

# Peak/off-peak distribution
peak_pct = random.uniform(0.3,0.4)
night_pct = 0.1
regular_pct = 1 - (peak_pct + night_pct)

peak_requests = int(num_requests * peak_pct)
night_requests = int(num_requests * night_pct)
regular_requests = num_requests - peak_requests - night_requests

requests = []
for period, n in [("peak",peak_requests),("regular",regular_requests),("night",night_requests)]:
    for _ in range(n):
        hour = random.choice(list(range(7,10)) + list(range(17,20))) if period=="peak" else \
               random.choice(list(range(22,24)) + list(range(0,5))) if period=="night" else \
               random.choice(list(set(range(24))-set(range(7,10))-set(range(17,20))-set(range(22,24))))
        minute = random.randint(0,59)
        request_time = datetime.now().replace(hour=hour,minute=minute,second=0,microsecond=0)
        src_lat, src_lon = 28.6 + random.uniform(-0.1,0.1), 77.2 + random.uniform(-0.1,0.1)
        dest_lat, dest_lon = 28.6 + random.uniform(-0.1,0.1), 77.2 + random.uniform(-0.1,0.1)
        est_time = random.randint(10,40)
        fare = est_time*random.uniform(3,6)
        driver_id = random.randint(1,len(drivers))
        requests.append((driver_id,"Delhi",request_time.isoformat(),src_lat,src_lon,dest_lat,dest_lon,est_time,fare,"requested"))
c.executemany("""
    INSERT INTO RideRequests (driver_id, city, requested_time, source_lat, source_lon, dest_lat, dest_lon, est_time, fare, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", requests)
conn.commit()

# -------------------------
# 4) SELECT COMPLETED RIDES
# -------------------------
# Randomly mark 1/3 as successful
df_requests = pd.read_sql("SELECT * FROM RideRequests", conn)
success_idx = df_requests.sample(num_successful).index
df_requests.loc[success_idx, "status"] = "completed"
df_requests.to_sql("RideRequests", conn, if_exists="replace", index=False)
conn.commit()

# -------------------------
# 5) VISUALIZE RIDES
# -------------------------
completed = df_requests[df_requests["status"]=="completed"]

plt.figure(figsize=(10,6))
plt.scatter(completed["source_lon"], completed["source_lat"], c='blue', alpha=0.5, label="Source")
plt.scatter(completed["dest_lon"], completed["dest_lat"], c='green', alpha=0.5, label="Destination")
plt.title(f"Completed Rides in Delhi ({len(completed)} rides)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()

# Timeline of requests
completed['requested_time'] = pd.to_datetime(completed['requested_time'])
completed['hour'] = completed['requested_time'].dt.hour
completed.groupby('hour').size().plot(kind='bar', figsize=(12,6), title='Completed Rides per Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Completed Rides')
plt.show()

conn.close()
'''

'''
# -------------------------
# This code is for generate maps for given number of days
# -------------------------
def simulate_day(day_num):
    conn = sqlite3.connect('uber_simulation.db')
    c = conn.cursor()

    # Reload drivers
    drivers_df = pd.read_sql("SELECT * FROM Drivers", conn)
    num_successful = int(500 * 1.2) if datetime.now().strftime('%A') in ['Monday','Friday','Saturday'] else 500
    num_requests = num_successful * 3

    peak_pct = random.uniform(0.3,0.4)
    night_pct = 0.1
    regular_pct = 1 - (peak_pct + night_pct)

    peak_requests = int(num_requests * peak_pct)
    night_requests = int(num_requests * night_pct)
    regular_requests = num_requests - peak_requests - night_requests

    requests = []
    for period, n in [("peak",peak_requests),("regular",regular_requests),("night",night_requests)]:
        for _ in range(n):
            if period == "peak":
                hour = random.choice(list(range(7,10)) + list(range(17,20)))
            elif period == "night":
                hour = random.choice(list(range(22,24)) + list(range(0,5)))
            else:
                excluded = set(range(7,10)) | set(range(17,20)) | set(range(22,24)) | set(range(0,5))
                regular_hours = [h for h in range(24) if h not in excluded]
                hour = random.choice(regular_hours)
            minute = random.randint(0,59)
            request_time = datetime.now().replace(hour=hour,minute=minute,second=0,microsecond=0)
            src_lat, src_lon = 28.6 + random.uniform(-0.1,0.1), 77.2 + random.uniform(-0.1,0.1)
            dest_lat, dest_lon = 28.6 + random.uniform(-0.1,0.1), 77.2 + random.uniform(-0.1,0.1)
            est_time = random.randint(10,40)
            fare = est_time*random.uniform(3,6)
            driver_id = random.randint(1,len(drivers_df))
            requests.append((driver_id,"Delhi",request_time.isoformat(),src_lat,src_lon,dest_lat,dest_lon,est_time,fare,"requested"))
    c.executemany("""
        INSERT INTO RideRequests (driver_id, city, requested_time, source_lat, source_lon, dest_lat, dest_lon, est_time, fare, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, requests)
    conn.commit()

    # Mark successful rides (1/3 of requests)
    df_requests = pd.read_sql("SELECT * FROM RideRequests", conn)
    success_idx = df_requests.sample(num_successful).index
    df_requests.loc[success_idx, "status"] = "completed"
    df_requests.to_sql("RideRequests", conn, if_exists="replace", index=False)
    conn.commit()

    # Update driver positions & idle times
    for _, ride in df_requests[df_requests["status"]=="completed"].iterrows():
        driver_id = ride["driver_id"]
        dest_lat, dest_lon = ride["dest_lat"], ride["dest_lon"]
        c.execute("UPDATE Drivers SET current_lat=?, current_lon=?, last_update_time=?, status='idle' WHERE driver_id=?",
                  (dest_lat, dest_lon, datetime.now().isoformat(), driver_id))

    # Return idle drivers to base if idle > 1hr
    drivers_df = pd.read_sql("SELECT * FROM Drivers", conn)
    now = datetime.now()
    for _, driver in drivers_df.iterrows():
        last_update = datetime.fromisoformat(driver["last_update_time"])
        if (now - last_update).total_seconds() > 3600:  # 1 hr
            # Move driver to base prime location
            prime = c.execute("SELECT latitude, longitude FROM PrimeLocations WHERE prime_id=?", (driver["base_prime_id"],)).fetchone()
            c.execute("UPDATE Drivers SET current_lat=?, current_lon=?, status='idle', last_update_time=? WHERE driver_id=?",
                      (prime[0], prime[1], datetime.now().isoformat(), driver["driver_id"]))
    conn.commit()

    # Generate folium interactive map of completed rides
    m = folium.Map(location=[28.6, 77.2], zoom_start=12)
    for _, ride in df_requests[df_requests["status"]=="completed"].iterrows():
        folium.Marker(location=[ride["source_lat"],ride["source_lon"]], popup=f"Src RideID {ride['request_id']}", icon=folium.Icon(color="blue")).add_to(m)
        folium.Marker(location=[ride["dest_lat"],ride["dest_lon"]], popup=f"Dest RideID {ride['request_id']}", icon=folium.Icon(color="green")).add_to(m)
        folium.PolyLine([(ride["source_lat"],ride["source_lon"]),(ride["dest_lat"],ride["dest_lon"])], color="orange", weight=1).add_to(m)
    m.save(f"day{day_num}_rides_map.html")

    # Cost analytics for the day
    df_completed = df_requests[df_requests["status"]=="completed"]
    df_completed["cost"] = (df_completed["est_time"]*0.00002 + 0.0001)  # simulate backend cost for each ride
    cost_per_driver = df_completed.groupby("driver_id")["cost"].sum().reset_index()
    print(f"Day {day_num} cost analytics:")
    print(cost_per_driver.head())

    conn.close()

# -------------------------------
# Scheduler: simulate multiple days
# -------------------------------
num_days = 3  # simulate 3 days
for day in range(1,num_days+1):
    print(f"\nSimulating Day {day}...")
    simulate_day(day)
    time.sleep(1)  # optional pause for realism
'''

def simulate_day(sim_day, sim_city):
    conn = sqlite3.connect('uber_simulation.db')
    c = conn.cursor()

    # Clear previous requests
    c.execute("DELETE FROM RideRequests")
    conn.commit()

    drivers_df = pd.read_sql("SELECT driver_id, shift_start, shift_end FROM Drivers", conn)

    # Check for missing shift assignments
    if drivers_df["shift_start"].isnull().any() or drivers_df["shift_end"].isnull().any():
        raise ValueError("🚨 Some drivers have missing shift_start or shift_end — please assign shifts before simulation!")

    # Determine how many rides to simulate today
    if sim_day in ['Monday', 'Friday', 'Saturday']:
        num_successful = random.randint(int(300 * 1.2), int(700 * 1.2))
    else:
        num_successful = random.randint(300, 700)
    num_requests = num_successful * 3

    peak_pct, night_pct = random.uniform(0.3, 0.4), 0.1
    regular_pct = 1 - (peak_pct + night_pct)
    peak_requests, night_requests = int(num_requests * peak_pct), int(num_requests * night_pct)
    regular_requests = num_requests - peak_requests - night_requests

    sim_date = datetime.now().date().isoformat()
    requests = []

    # Build active hour dictionary per driver
    driver_hours = {}
    for _, d in drivers_df.iterrows():
        start, end = d["shift_start"], d["shift_end"]
        if end <= 24:
            active = list(range(start, end))
        else:
            active = list(range(start, 24)) + list(range(0, end - 24))
        driver_hours[d["driver_id"]] = active

    # Helper: get available drivers at a specific hour
    def get_available_drivers(req_hour):
        return [
            did for did, hours in driver_hours.items()
            if req_hour in hours
        ]

    for period, n in [("peak", peak_requests), ("regular", regular_requests), ("night", night_requests)]:
        for _ in range(n):
            if period == "peak":
                possible_hours = list(set(range(7, 10)) | set(range(17, 20)))
            elif period == "night":
                possible_hours = list(set(range(22, 24)) | set(range(0, 5)))
            else:
                excluded = set(range(7, 10)) | set(range(17, 20)) | set(range(22, 24)) | set(range(0, 5))
                possible_hours = [h for h in range(24) if h not in excluded]

            hour = random.choice(possible_hours)
            minute = random.randint(0, 59)
            request_time = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
            request_hour = hour % 24  # wrap-around safety

            available_drivers = get_available_drivers(request_hour)
            if not available_drivers:
                continue  # skip request if no driver available at this hour

            driver_id = random.choice(available_drivers)
            src_lat, src_lon = 28.6 + random.uniform(-0.1, 0.1), 77.2 + random.uniform(-0.1, 0.1)
            dest_lat, dest_lon = 28.6 + random.uniform(-0.1, 0.1), 77.2 + random.uniform(-0.1, 0.1)
            est_time = random.randint(10, 40)
            distance = est_time * random.uniform(0.3, 0.6)
            fare = 0.5 + distance * 0.108 + est_time * 0.018
            requests.append((driver_id, sim_city, sim_date, request_time.isoformat(),
                             src_lat, src_lon, dest_lat, dest_lon, est_time, fare, "requested"))

    if not requests:
        print("🚨 No ride requests generated — skipping ride completion step.")
        return

    c.executemany("""
        INSERT INTO RideRequests (driver_id, city, sim_date, requested_time, source_lat, source_lon, dest_lat, dest_lon, est_time, fare, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, requests)
    conn.commit()

    df_requests = pd.read_sql("SELECT * FROM RideRequests", conn)

    if len(df_requests) < num_successful:
        print(f"⚠️ Not enough requests ({len(df_requests)}) to sample {num_successful} successful rides. Adjusting num_successful...")
        num_successful = len(df_requests)

    if num_successful > 0:
        success_idx = df_requests.sample(num_successful).index
        df_requests.loc[success_idx, "status"] = "completed"
    else:
        print("🚨 No ride requests generated — skipping ride completion step.")

    if len(requests) < 50:
        print("⚠️ Very few requests generated — consider adjusting driver shifts or request distribution.")

    df_requests.to_sql("RideRequests", conn, if_exists="replace", index=False)
    conn.commit()

    # Update driver locations for completed rides
    for _, ride in df_requests[df_requests["status"] == "completed"].iterrows():
        c.execute("UPDATE Drivers SET current_lat=?, current_lon=?, last_update_time=?, status='idle' WHERE driver_id=?",
                  (ride["dest_lat"], ride["dest_lon"], datetime.now().isoformat(), ride["driver_id"]))

    # Return idle drivers to base if idle > 1 hour
    drivers_df = pd.read_sql("SELECT * FROM Drivers", conn)
    now = datetime.now()
    for _, driver in drivers_df.iterrows():
        last_update = datetime.fromisoformat(driver["last_update_time"])
        if (now - last_update).total_seconds() > 3600:
            prime = c.execute("SELECT latitude, longitude FROM PrimeLocations WHERE prime_id=?",
                              (driver["base_prime_id"],)).fetchone()
            if prime:
                c.execute("UPDATE Drivers SET current_lat=?, current_lon=?, status='idle', last_update_time=? WHERE driver_id=?",
                          (prime[0], prime[1], now.isoformat(), driver["driver_id"]))
    conn.commit()

    # Generate folium interactive map of completed rides
    m = folium.Map(location=[28.6, 77.2], zoom_start=12)
    for _, ride in df_requests[df_requests["status"] == "completed"].iterrows():
        folium.Marker(location=[ride["source_lat"], ride["source_lon"]], popup=f"Src RideID {ride['request_id']}", icon=folium.Icon(color="blue")).add_to(m)
        folium.Marker(location=[ride["dest_lat"], ride["dest_lon"]], popup=f"Dest RideID {ride['request_id']}", icon=folium.Icon(color="green")).add_to(m)
        folium.PolyLine([(ride["source_lat"], ride["source_lon"]), (ride["dest_lat"], ride["dest_lon"])], color="orange", weight=1).add_to(m)
    m.save(f"{sim_day}_{sim_city}_rides_map.html")

    # Cost analytics for the day
    df_completed = df_requests[df_requests["status"] == "completed"].copy()
    df_completed["driver_id"] = df_completed["driver_id"].apply(
        lambda x: int.from_bytes(x, byteorder="little") if isinstance(x, bytes) else x)
    df_requests["driver_id"] = df_requests["driver_id"].apply(
        lambda x: int.from_bytes(x, byteorder="little") if isinstance(x, bytes) else x
    )
    completed_mask = df_requests["status"] == "completed"
    processing_mask = df_requests["status"] != "completed"
    df_requests.loc[completed_mask, "cost"] = 0.02376 * np.random.uniform(0.9, 1.1, completed_mask.sum())
    df_requests.loc[processing_mask, "cost"] = 0.02376 * 0.3 * np.random.uniform(0.9, 1.1, processing_mask.sum())

    total_cost = df_requests["cost"].sum()
    avg_cost = df_requests["cost"].mean()
    cost_per_driver = df_completed.groupby("driver_id")["cost"].sum().reset_index()
    df_requests.to_sql("RideRequests", conn, if_exists="replace", index=False)
    conn.commit()
    #print(f"{sim_day} in {sim_city} cost analytics:")
    #print(cost_per_driver.head())
    #print(f"Total backend cost: ${total_cost:.2f}")
    #print(f"Average backend cost per ride: ${avg_cost:.5f}")

    conn.close()


