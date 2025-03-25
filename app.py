import streamlit as st
import pandas as pd
import plotly.express as px
import random
from google.cloud import bigquery
import os
from dbharbor.bigquery import SQL
import streamlit as st
import pandas as pd
from google.cloud import bigquery
import matplotlib.pyplot as plt
from dbharbor.bigquery import SQL
import os
from dotenv import load_dotenv
import pathlib
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime,date,timedelta
import pytz  # if using Python <3.9; otherwise use zoneinfo
import numpy as np

from authlib.integrations.requests_client import OAuth2Session
  
# Load environment variables
load_dotenv()




# Google OAuth setup - You'll need to replace these with your credentials
client_id =  os.getenv('GOOGLE_CLIENT_ID') 
client_secret = os.getenv('GOOGLE_CLIENT_SECRET') 
redirect_uri = os.getenv('REDIRECT_URI') 

# OAuth2.0 authorization URL for Google
authorization_url = "https://accounts.google.com/o/oauth2/auth"
token_url = "https://accounts.google.com/o/oauth2/token"
api_base_url = "https://www.googleapis.com/oauth2/v1"

# Create the OAuth2Session object
client = OAuth2Session(client_id, client_secret, redirect_uri=redirect_uri)

# Define the scope of permissions you need (in this case, basic user info)
scope = ['https://www.googleapis.com/auth/userinfo.profile', 'https://www.googleapis.com/auth/userinfo.email']

# Generate the authorization URL and state
authorization_url, state = client.create_authorization_url(
    authorization_url,
    scope=scope,
    access_type='offline',
    prompt='consent'
)



st.set_page_config(page_title="📊 Job Tracker", page_icon="📈", layout="wide")

query_params = st.query_params

# Handle redirect from Google
if "code" in query_params:
    code = query_params["code"][0]
    
    try:
        token = client.fetch_token(token_url, code=code)
        st.session_state.token = token
        st.session_state.authenticated = True
        
        user_info = client.get(f"{api_base_url}/userinfo").json()
        st.session_state.user_info = user_info
        
        st.success(f"Welcome {user_info['name']}!")
    except Exception as e:
        st.error("Authentication failed.")
        st.stop()

# If not authenticated yet, show login
elif not st.session_state.get("authenticated", False):
    st.markdown(f"Please log in with Google:")
    st.markdown(f"[Login with Google]({authorization_url})")
    st.stop()

# Already authenticated
else:
    user_info = st.session_state.get("user_info", {})
    st.write(f"Welcome back, {user_info.get('name', 'User')}!")



# New Color Palette
PRIMARY_COLOR = "#2E86C1"  # Soft Trustworthy Blue
SECONDARY_COLOR = "#28B463"  # Safe Green
TABLE_HEADER_BG = "#AED6F1"  # Light Blue (Homely, Safe)
TEXT_COLOR = "#212121"  # Dark Gray

#  Custom CSS for Sidebar & Table Styling
st.markdown(f"""
    <style>
        body {{
            background-color: #F8F9FA;
        }}
        .block-container {{
            padding: 2rem;
            background: white;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }}
        h1, h2, h3, h4 {{
            color: {TEXT_COLOR};
            text-align: center;
        }}
        .stMetric {{
            text-align: center;
        }}
        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background-color: {PRIMARY_COLOR};
            color: white;
        }}

        /* Make Sidebar Title ( Navigation) White */
        [data-testid="stSidebar"] h1 {{
            color: white !important;
            font-weight: bold !important;
            font-size: 24px !important;
        }}


        [data-testid="stSidebarNav"] > div {{
            color: white;
        }}

        /* Change Multi-Select Dropdown Text to White & Bold in Sidebar */
        [data-testid="stSidebar"] .stMultiSelect label {{
            color: white !important;  /* Makes the label text white */
            font-weight: bold !important;  /* Makes it bold */
            font-size: 16px !important;  /* Slightly larger for readability */
        }}

        [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] {{
            color: white !important;  /* Changes dropdown text to white */
            font-weight: bold !important;
            font-size: 16px !important;

        }}



        span[data-baseweb="tag"] {{
            background-color: #585e60 !important;  /* Change this to any color */
            color: white !important;  /* Text color */
            border-radius: 8px !important;  /* Rounded corners for smoother look */
            padding: 5px 10px !important;
            font-weight: bold !important;
        }}



        /* Table Header Styling */
        thead th {{
            background-color: {TABLE_HEADER_BG} !important;
            color: black !important;
            font-size: 16px !important;
            text-align: center !important;
        }}
        /* Centered Table */
        .centered-table {{
            width: 70%;
            margin: auto;
            text-align: center;
        }}

        .centered-table2 {{
            display: flex;
            justify-content: center;
        }}
    </style>
""", unsafe_allow_html=True)


st.markdown(
    """
    <style>
        /* Make all sidebar headings white */
        .sidebar-title, .sidebar-subheading {
            color: white !important;
            font-weight: bold;
            font-size: 18px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
        /* Make all sidebar labels white */
        div[data-testid="stDateInput"] label {
            color: white !important;
            font-weight: bold;
            font-size: 16px;
        }
    </style>
    """,
    unsafe_allow_html=True
)



@st.cache_data(ttl=60 * 5)
def fetch_data_from_bigquery(start_date, end_date):
    con = SQL(credentials_filepath=os.getenv('BIGQUERY_CRED'))
    
    sql = f"""

        WITH successful_runs AS (
            SELECT 
                job_name,
                job_id,
                created_at,
                ended_at,
                source_system,
                status,
                DATETIME_DIFF(COALESCE(ended_at, CURRENT_DATETIME()), created_at, MINUTE) AS duration_mins,
                error_message
            FROM bbg-platform.analytics.job_statuses
            WHERE LOWER(status) IN ('successful', 'success', 'succeeded')
            QUALIFY ROW_NUMBER() OVER (PARTITION BY job_name ORDER BY created_at DESC) <= 10

        ),
        average_success_duration as
        (
        select job_name,avg(duration_mins) avg_duration_mins from successful_runs  group by all 
        )

        SELECT 
              j.job_name AS `Job Name`,
              job_id AS `Job Id`,
              created_at AS `Start Time`,
              ended_at AS `End Time`,
              source_system AS `Tool`,
              CASE
                  WHEN LOWER(status) IN ('successful', 'success','succeeded') THEN 'Successful'
                  WHEN LOWER(status) IN ('failed', 'failure', 'rescheduled', 'error','warning') THEN 'Failed'
                  WHEN LOWER(status) IN ('running', 'in_progress', 'processing', 'querying') THEN 'Running'
                  WHEN LOWER(status) IN ('cancelled') THEN 'Cancelled'
              END AS Status,
              DATETIME_DIFF(COALESCE(ended_at, CURRENT_DATETIME()), created_at, MINUTE) AS `Duration mins`,
              avg_duration_mins,
              error_message
        FROM bbg-platform.analytics.job_statuses j
        left join average_success_duration a on a.job_name = j.job_name
        WHERE DATE(created_at) BETWEEN DATE('{start_date}') AND DATE('{end_date}')
    """

    return con.read(sql)


def format_number(n):
    if n < 1000:
        return str(n)
    elif n < 1000000:
        return f"{n / 1000:.2f}k".rstrip("0").rstrip(".")
    else:
        return f"{n / 1000000:.2f}M".rstrip("0").rstrip(".")
# def generate_dummy_data(n=100):
#     job_data = []
#     for i in range(n):
#         tool = random.choice(TOOL_OPTIONS)
#         status = random.choice(STATUS_OPTIONS)
#         duration = random.uniform(1, 100) if status == "Success" else random.uniform(1, 200)
#         start_time = pd.Timestamp.now() - pd.Timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))
#         end_time = start_time + pd.Timedelta(minutes=duration) if status == "Success" else None
#         job_data.append([f"Job-{i+1}", tool, status, round(duration, 2), start_time, end_time])
#     return pd.DataFrame(job_data, columns=["Job Name", "Tool", "Status", "Duration (mins)", "Start Time", "End Time"])


def highlight_failed(row):
    if row["Status"] == "Failed":
        return ["background-color: #ffe6e6; color: red"] * len(row)
    return [""] * len(row)
# Timezone for Arizona
ARIZONA_TZ = pytz.timezone("America/Phoenix")

# Get current datetime in Arizona
now_az = datetime.now(ARIZONA_TZ)
today_az = now_az.date()
default_start = today_az - timedelta(days=30)
default_end = today_az

df = fetch_data_from_bigquery(default_start,default_end)

#  Sidebar Navigation
with st.sidebar:
    st.title("🔍 Navigation")


    # Sidebar Job Name Filter
    st.markdown('<h3 class="sidebar-subheading">🏗️ Filter by Job Name</h3>', unsafe_allow_html=True)

    job_names = sorted(df["Job Name"].unique().tolist())
    all_label = "All Jobs"

    # UI: include 'All Jobs' option
    job_options = [all_label] + job_names

    # Default: just show 'All Jobs' selected
    selected_jobs = st.multiselect(
        "Search & Select Jobs",
        job_options,
        default=[all_label],
        help="Start typing to find a job quickly"
    )

    # Logic: if 'All Jobs' is selected or nothing is selected, include all jobs
    if all_label in selected_jobs:
        filtered_jobs = job_names
    else:
        filtered_jobs = selected_jobs


    st.session_state.selected_jobs = selected_jobs
    #  Fixed Tool & Status Lists
    TOOL_OPTIONS = sorted(df["Tool"].unique().tolist())
    STATUS_OPTIONS = sorted(["Running" if s is None else s for s in df["Status"].unique().tolist()])



    selected_tool = st.multiselect("Filter by Tool", TOOL_OPTIONS, default=TOOL_OPTIONS)
    selected_status = st.multiselect("Filter by Status", STATUS_OPTIONS, default=STATUS_OPTIONS)
    #  Date Filter (From - To)


    st.markdown('<h3 class="sidebar-subheading">📅 Filter by Date</h3>', unsafe_allow_html=True)

    date_range = st.date_input(
        "Select Date Range (Arizona Time)",
        value=[default_start, default_end],
        min_value=default_start - timedelta(days=365),
        max_value=default_end
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        st.stop()  # prevent fetch if invalid



    st.markdown("---")
    st.write("💡 **Tip:** Use filters to explore trends!")

df_filtered = df[df["Tool"].isin(selected_tool) & df["Status"].isin(selected_status)
                    & df["Job Name"].isin(filtered_jobs)
                    & (df["Start Time"].dt.date >= start_date) 
                    & (df["Start Time"].dt.date <= end_date)
                 ]

#  Dashboard Title
st.markdown(f"<h1>📊 Job Execution Tracker</h1>", unsafe_allow_html=True)

# Centered Metrics Section
st.markdown("<h2 style='text-align: center;'>📌 Summary</h2>", unsafe_allow_html=True)
col1, col2, col3, col4,col5 = st.columns([1,1, 1, 1,2]) 

with col2:
    df_len = format_number(len(df_filtered))
    st.metric(label="📊 Total Runs", value=f"{df_len}")

with col3:
    df_len_success = len(df_filtered[df_filtered["Status"] == "Successful"])
    df_len_success_format = format_number(df_len_success)
    st.metric(label="✅ Successes", value= df_len_success_format)

with col4:
    df_len_failure = len(df_filtered[df_filtered["Status"] == "Failed"])
    df_len_failure_format = format_number(df_len_failure)
    st.metric(label="❌ Failures", value= df_len_failure_format)

with col5:
    success_rate = (len(df_filtered[df_filtered["Status"] == "Successful"]) / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
    st.metric(label="🎯 Success Rate", value=f"{success_rate:.2f}%")


#  Recent Runs Table (Centered)
st.markdown("<h2 style='text-align: center;'>📝 Recent Job Runs</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 4, 1])  # Creates three columns, middle one is larger
with col2:
    df_filtered.loc[
        (df_filtered["Status"] == "Failed") & (df_filtered["error_message"].isna()),
        "error_message"
    ] = "Operation did not complete within the designated timeout of 300 seconds"
    top_50 = df_filtered.sort_values(by="Start Time", ascending=False).head(50)
    styled_df = top_50.style.apply(highlight_failed, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


#  Failed Runs Table (Centered)
st.markdown("<h2 style='text-align: center; color: red;'>❌ Failed Runs</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 4, 1])  # Creates three columns, middle one is larger
with col2:
    st.dataframe(df_filtered[df_filtered["Status"] == "Failed"].sort_values("Start Time",ascending = False)
                 .head(100),hide_index=True)



col1, col2, col3 = st.columns([1, 4, 1])  # Creates three columns, middle one is larger
with col2:
    # Calculate ratio of current run duration to average
# Compute ratio, but set to NaN if avg_duration_mins is 0 or NaN
    df["Duration Ratio"] = np.where(
        df["avg_duration_mins"].fillna(0) == 0,
        np.nan,
        df["Duration mins"] / df["avg_duration_mins"]
    )
    # Filter: Jobs that took at least 2x their normal average
    df_spiked = df[(df["Duration Ratio"] >= 2) & (df["Status"] == 'Running')].copy()

    # Sort by highest deviation
    df_spiked = df_spiked.sort_values(by="Duration Ratio", ascending=False)


    # Display top offenders
    st.markdown("<h2 style='text-align: center;'>🐢 Jobs Running Twice as Long</h2>", unsafe_allow_html=True)
    st.dataframe(
        df_spiked[
            ["Job Name", "Tool", "Status", "Duration mins", "avg_duration_mins", "Duration Ratio"]
        ],
        use_container_width=True,
        hide_index=True
    )


#  Job Execution Time Trend
st.markdown(f"<h2 style='text-align: center;'>📈 Job Duration Trend Over Time</h2>", unsafe_allow_html=True)
df_successful = df_filtered[df_filtered["Status"].isin(["Successful", "Running"])].sort_values("Start Time")

if not df_successful.empty:
    fig = px.line(df_successful, x="Start Time", y="Duration mins", color="Tool", markers=True,
                  title="Job Execution Time Trend",
                  color_discrete_sequence=["#1E88E5", "#F4511E", "#43A047", "#8E24AA", "#FB8C00"],
                  hover_data=["Job Name"]
                  )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No successful jobs available to display trend.")



# Extract the date
df_filtered["Date"] = df_filtered["Start Time"].dt.date

# Create a binary success flag
df_filtered["Is Failed"] = df_filtered["Status"] == "Failed"

# Group by Date and Tool, and calculate success rate
failure_trend = (
    df_filtered.groupby(["Date", "Tool"])
    .agg(
        failure_count=("Is Failed", "sum"),
        total_count=("Status", "count")
    )
    .reset_index()
)


failure_trend["Failure Rate"] = (failure_trend["failure_count"] / failure_trend["total_count"]).round(2)

# Plot the trend
fig = px.line(
    failure_trend,
    x="Date",
    y="Failure Rate",
    color="Tool",
    title="📈 Average Daily Failure Rate per Tool",
    markers=True,
    labels={"Failure Rate": "Failure Rate (%)"}
)

fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])

st.markdown("<h2 style='text-align: center;'>📈 Average Daily Failure Rate per Tool</h2>", unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True)

# Longest Running Jobs (Centered)
st.markdown(f"<h2 style='text-align: center;'> Current Running Jobs</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 3, 1])  # Creates three columns, middle one is larger
with col2:
    st.dataframe(df_filtered[df_filtered["Status"] == "Running"].sort_values("Start Time",ascending = False)
                 .head(100),hide_index=True)

# Most Failing Jobs (Top 5)
st.markdown("<h2 style='text-align: center;'>❌ Most Failing Jobs (Top 10)</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1.5, 2, 1])  # Creates three columns, middle one is larger
with col2:
    df_failures = df_filtered[df_filtered["Status"] == "Failed"]
    fail_counts = df_failures["Job Name"].value_counts().reset_index().rename(columns={"index": "Job Name", "Job Name": "Job Name"})
    st.dataframe(fail_counts.head(10),hide_index=True)

#  Job Execution Frequency (Heatmap)
st.markdown("<h2 style='text-align: center;'>🔥 Job Execution Frequency (Heatmap)</h2>", unsafe_allow_html=True)
df_filtered["Hour"] = df_filtered["Start Time"].dt.hour
heatmap_data = df_filtered.groupby(["Hour", "Tool"]).size().reset_index(name="Count")

fig = px.density_heatmap(heatmap_data, x="Hour", y="Tool", z="Count", color_continuous_scale="blues", title="Job Execution Frequency by Hour")
st.plotly_chart(fig, use_container_width=True)

#  All Jobs List
#  Job-Level Statistics Table (Summarized Per Job)
st.markdown("<h2 style='text-align: center;'>📊 Job-Level Statistics</h2>", unsafe_allow_html=True)

# Group by Job Name and Aggregate
job_stats = df_filtered.groupby("Job Name").agg(
    Total_Runs=("Status", "count"),
    Success_Count=("Status", lambda x: (x == "Successful").sum()),
    Failure_Count=("Status", lambda x: (x == "Failed").sum()),
    Avg_Duration=("Duration mins", "mean"),  # Fixed this field
    First_Run=("Start Time", "min"),
    Last_Run=("Start Time", "max")
).reset_index()

# Convert timestamps to readable format
job_stats["First_Run"] = job_stats["First_Run"].dt.strftime("%Y-%m-%d %H:%M:%S")
job_stats["Last_Run"] = job_stats["Last_Run"].dt.strftime("%Y-%m-%d %H:%M:%S")

col1, col2, col3 = st.columns([1, 4, 1])  # Creates three columns, middle one is larger
with col2:
    # Sort jobs by total runs (most frequent jobs at the top)
    st.dataframe(job_stats.sort_values(by="Total_Runs", ascending=False),hide_index = True)



# Clear cache on browser refresh
if "app_initialized" not in st.session_state:
    # First load of the app or browser refresh
    st.session_state.app_initialized = True
    st.cache_data.clear()  # Clear cache automatically on browser refresh
    
#  Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>🚀 For the Data Analytics Team | MASTERMIND ✨</p>", unsafe_allow_html=True)

