import streamlit as st
import sqlite3
import pandas as pd
import folium
from streamlit_folium import st_folium
import time
import ryde


# -------------------
# 🔧 Simulation Controls
# -------------------
st.sidebar.title("🛠️ Run New Simulation")

city_options = ["Delhi"]  # You can dynamically fetch this from your Cities table if needed
sim_city = st.sidebar.selectbox("Select City", city_options)

sim_day = st.sidebar.selectbox("Select Simulation Day", [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
])

if st.sidebar.button("🚀 Run Simulation"):
    with st.spinner(f"Running simulation for {sim_city} on {sim_day}..."):
        ryde.simulate_day(sim_day, sim_city)
        st.success(f"Simulation complete for {sim_city} on {sim_day}!")
        st.rerun()

# Connect to the simulation database
conn = sqlite3.connect("uber_simulation.db")

# Load data
df_requests = pd.read_sql("SELECT * FROM RideRequests", conn)
df_requests["driver_id"] = df_requests["driver_id"].apply(
    lambda x: int.from_bytes(x, byteorder="little") if isinstance(x, bytes) else x
)

df_drivers = pd.read_sql("SELECT * FROM Drivers", conn)
df_drivers["driver_id"] = df_drivers["driver_id"].apply(
    lambda x: int.from_bytes(x, byteorder="little") if isinstance(x, bytes) else x
)
df_requests["cost"] = pd.to_numeric(df_requests["cost"], errors="coerce")

# Main KPIs
total_requests = len(df_requests)
total_completed = len(df_requests[df_requests["status"]=="completed"])

df_completed = df_requests[df_requests["status"] == "completed"].copy()
driver_earnings = df_completed.groupby("driver_id")["fare"].sum()
avg_driver_earning = driver_earnings.mean() if not driver_earnings.empty else 0
rides_per_driver = df_completed.groupby("driver_id").size()
avg_rides_per_driver = rides_per_driver.mean() if not rides_per_driver.empty else 0
avg_fare_per_ride = df_completed["fare"].mean() if not df_completed.empty else 0
if "cost" in df_requests.columns:
    total_cost = df_requests["cost"].sum(skipna=True)
    avg_cost = df_requests["cost"].mean(skipna=True) if total_requests > 0 else 0
else:
    total_cost = 0
    avg_cost = 0

# Convert to INR
USD_TO_INR = 83
total_cost_inr = total_cost * USD_TO_INR
avg_cost_inr = avg_cost * USD_TO_INR
avg_driver_earning_inr = avg_driver_earning * USD_TO_INR
avg_fare_per_ride_inr = avg_fare_per_ride * USD_TO_INR

st.title("🚦 Ryde Simulation Dashboard")
st.markdown("---")

if st.button("🔄 Refresh Data"):
    st.rerun()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Ride Requests", f"{total_requests:,}")
col2.metric("Completed Rides", f"{total_completed:,}")
col3.metric("Est. Backend Cost (₹)", f"₹{total_cost_inr:,.3f}")
col4.metric("Avg Backend Cost/Ride (₹)", f"₹{(total_cost_inr/total_requests if total_completed else 0):,.4f}")
col5, col6, col7 = st.columns(3)
col5.metric("Avg Driver Earnings (₹)", f"₹{avg_driver_earning_inr:,.2f}")
col6.metric("Avg Rides/Driver", f"{avg_rides_per_driver:.2f}")
col7.metric("Avg Fare per Ride (₹)", f"₹{avg_fare_per_ride_inr:,.2f}")

st.markdown("## 📊 Ride Distribution")

# Ride completion over time
df_requests["requested_time"] = pd.to_datetime(df_requests["requested_time"])
df_requests["hour"] = df_requests["requested_time"].dt.hour
completed_per_hour = df_requests[df_requests["status"]=="completed"].groupby("hour").size()

st.bar_chart(completed_per_hour)

def categorize_hour(hour):
    if 7 <= hour < 10 or 17 <= hour < 20:
        return "Peak"
    elif hour in list(range(22,24)) + list(range(0,5)):
        return "Night"
    else:
        return "Regular"

df_requests["period"] = df_requests["hour"].apply(categorize_hour)
period_counts = df_requests.groupby("period").size()

st.markdown("## 📊 Ride Period Distribution")
st.bar_chart(period_counts)

# Rides per driver
rides_per_driver = df_requests[df_requests["status"]=="completed"].groupby("driver_id").size()
st.bar_chart(rides_per_driver)

# Interactive map
st.markdown("## 🗺️ Completed Rides Map")

m = folium.Map(location=[28.6, 77.2], zoom_start=12)
completed = df_requests[df_requests["status"]=="completed"]
for _, ride in completed.iterrows():
    folium.Marker([ride["source_lat"],ride["source_lon"]],
                  popup=f"Request {ride['request_id']} Source",
                  icon=folium.Icon(color='blue')).add_to(m)
    folium.Marker([ride["dest_lat"],ride["dest_lon"]],
                  popup=f"Request {ride['request_id']} Dest",
                  icon=folium.Icon(color='green')).add_to(m)
    folium.PolyLine([(ride["source_lat"],ride["source_lon"]), (ride["dest_lat"],ride["dest_lon"])],
                    color="orange", weight=1).add_to(m)

st_data = st_folium(m, width=700, height=500)

if st.sidebar.checkbox("Enable Day-by-Day Analytics"):
    daily_summary = df_requests[df_requests["status"]=="completed"].groupby(df_requests["requested_time"].dt.date).size()
    st.line_chart(daily_summary)

st.markdown("## 📈 Completed Rides Per Day")
rides_per_day = df_requests[df_requests["status"]=="completed"].groupby("sim_date").size()
st.line_chart(rides_per_day)

# Filters
st.sidebar.title("🔎 Filters")
selected_city = st.sidebar.selectbox("Filter by City", ["All"] + city_options)
if selected_city != "All":
    df_requests = df_requests[df_requests["city"] == selected_city]

# Determine date range based on sim_date (not requested_time anymore)
date_options = sorted(df_requests["sim_date"].unique())
if not date_options:
    st.error("No data found in RideRequests. Please run simulation first.")
    st.stop()

selected_dates = st.sidebar.multiselect(
    "Select Simulation Dates",
    options=date_options,
    default=date_options[-1:]  # show the latest by default
)

# Filter df_requests by selected dates
df_requests = df_requests[df_requests["sim_date"].isin(selected_dates)]

st.markdown(f"### Showing data for dates: {', '.join(selected_dates)}")

driver_options = df_drivers["driver_name"].unique()
selected_driver = st.sidebar.selectbox("Select Driver", options=["All"] + list(driver_options))
if selected_driver != "All":
    driver_id = df_drivers[df_drivers["driver_name"]==selected_driver]["driver_id"].values[0]
    driver_rides = df_requests[df_requests["driver_id"]==driver_id]
    st.write(f"### Rides for {selected_driver}")
    st.dataframe(driver_rides)


conn.close()
#df_requests.to_csv("RideRequests.csv", index=False)


