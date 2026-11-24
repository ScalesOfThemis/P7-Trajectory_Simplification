from pathlib import Path
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
#todo: Add error/loss between simplified and original

#Input folder names for original and simplified trajectories, make sure trajectories have same format and names
original_name = "Geolife_out"
simplified_name = "Geolife_out"


def load_data(folder):
    trajectories = {}

    for file in Path(folder).iterdir():
        if file.is_file():
            trajectories[file.name] = np.loadtxt(file)

    return trajectories

st.set_page_config(layout="wide")

st.title("Trajectory Viewer")

orig_trajectories = load_data(original_name)
simp_trajectories = load_data(simplified_name)

col1, col2 = st.columns(2)

selected_name = st.selectbox(
    "Select trajectory",
    list(orig_trajectories.keys())
)

original = orig_trajectories[selected_name]
simplified = simp_trajectories[selected_name]

with col1:
    fig1, ax1 = plt.subplots()

    ax1.plot(
        original[:, 0],  # longitude
        original[:, 1] ,  # latitude
        label="original"
    )

    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_title("Original")
    ax1.legend()

    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()

    ax2.plot(
        simplified[:, 0],  # longitude
        simplified[:, 1], # latitude
        label="simplified"
    )

    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_title("Simplified")
    ax2.legend()

    st.pyplot(fig2)