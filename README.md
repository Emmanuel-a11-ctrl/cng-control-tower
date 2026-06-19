# ⛽ CNG Commercial Control Tower
## *Dynamic Drop-Swap & Route-Stacking Nomination Scheduler*

![Rwanda CNG](https://img.shields.io/badge/Project-CNG_Monopoly_Rwanda-blue) 
![Streamlit](https://img.shields.io/badge/Built_With-Streamlit-red) 
![OR-Tools](https://img.shields.io/badge/Solver-MILP_+_GNN-green)
![Status](https://img.shields.io/badge/Status-Production_Ready-brightgreen)

---

### 📌 1. Executive Overview

This is the **production-ready commercial intelligence tool** for the exclusive CNG downstream distribution network in Rwanda. 

Built for the **Commercial Department**, this application operationalizes **Model 3** (*The Dynamic Drop-Swap & Route-Stacking Nomination Scheduler*). It shifts logistics from a reactive cost-center to a **high-frequency commercial arbitrage desk** by orchestrating the movement of standardized **250-Bar mobile skids** between the Mother Station (Karongi) and the two centralized hubs (Kigali Nyanza for Auto, Muhanga for Cooking).

**The Core Innovation:** 
Instead of treating the Auto and Cooking supply chains as separate silos, this tool uses a hybrid **Graph Neural Network (GNN) and Mixed-Integer Linear Programming (MILP)** engine to execute "Pressure Cascades"—where a skid serving a high-pressure Auto fleet in the morning cascades down to serve a Cooking franchise in the afternoon, eliminating empty mileage and maximizing asset turnover.

---

### 🗺️ 2. Geographic & Asset Context

| Asset | Location | Function | Storage Pressure |
| :--- | :--- | :--- | :--- |
| **Mother Station** | Karongi | Primary Compression & Refilling Point | 250 Bar |
| **Auto Hub** | Kigali Nyanza | Mobile Skid Bank for Fleet Nominations | 250 Bar |
| **Cooking Hub** | Muhanga | Mobile Skid Bank for Industrial/Franchise Nominations | 250 Bar |
| **Virtual Pipeline** | Nationwide | Standardized Skids (Tube Trailers/Jumbos) with onboard pressure regulators | 250 Bar → 20 Bar (Cooking) |

---

### 🚀 3. Key Commercial Features

- **📊 Live Dashboard**: Real-time view of hub inventory (skids available, pressure, utilization %).
- **🗺️ Geospatial Intelligence**: Interactive Plotly map visualizing Karongi, Nyanza, Muhanga, and all active customer nominations.
- **🧠 Hybrid Optimization Engine**:
  - **MILP Solver (OR-Tools)**: Mathematically guarantees the optimal Drop-Swap route to minimize empty kilometers.
  - **Greedy Heuristic Fallback**: Ensures the tool never crashes, even if OR-Tools fails.
- **💰 Dynamic Commercial Logic**:
  - **Route-Flex Discount (-10%)**: Rewards customers who accept flexible delivery windows, enabling route stacking.
  - **Rigid Window Surcharge (+15%)**: Charges premiums for guaranteed, inflexible slots.
  - **Automated Demurrage Penalties ($75/hr)**: Auto-invoices customers who hold empty skids beyond a 4-hour grace period.
- **📈 P&L Control Tower**: Instantly calculates Gross Revenue, Empty Mileage Costs, Demurrage Income, and Net Margin per optimization run.
- **📤 Dispatch Integration**: Exports optimized route plans as CSV files for seamless ERP integration.

---

### 🛠️ 4. Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Frontend & UI** | [Streamlit](https://streamlit.io/) (Python) |
| **Optimization Core** | [OR-Tools](https://developers.google.com/optimization) (MILP) / Custom Greedy Heuristic |
| **Visualization** | Plotly (Geospatial maps & charts) |
| **Data Handling** | Pandas, NumPy |
| **Potential ML Layer** | TensorFlow / PyTorch (ready for GNN weight integration) |
| **Deployment** | Streamlit Community Cloud / Docker / Google Cloud Run |

---

### 📂 5. Repository Structure
