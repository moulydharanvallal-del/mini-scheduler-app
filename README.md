# 🔁 FlowLab

**Scheduling, Sequencing, and Routing Logic for Manufacturing Systems**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mini-scheduler-app.streamlit.app)

---

## What is FlowLab?

FlowLab is an open educational platform that exposes how manufacturing scheduling actually works. It implements a backward, constraint-aware scheduling engine that reasons through routing logic, precedence constraints, and capacity limits to construct feasible schedules.

**This is a tool for thinking, not button-clicking.**

---

## ✨ Features

- **📋 Order Management** — Define customer orders with products, quantities, and due dates
- **🔧 Bill of Materials** — Configure multi-level BOMs with routing steps and cycle times
- **🏭 Work Center Capacity** — Set machine availability at each work center
- **📊 Visual Scheduling** — Interactive Gantt charts showing job flow over time
- **📈 Capacity Analysis** — Utilization charts and bottleneck identification
- **🗺️ Routing Maps** — Visual BOM/routing diagrams
- **📥📤 CSV Import/Export** — Edit data in Excel, upload to run

---

## 🚀 Quick Start

1. Visit [mini-scheduler-app.streamlit.app](https://mini-scheduler-app.streamlit.app)
2. Review sample data in Orders, BOM, and Work Centers tabs
3. Click **Run Scheduler** in the sidebar
4. Explore results in the Results tab

Or: Download CSVs → Edit in Excel → Upload → Run

---

## 🧩 What You Can Explore

- Backward scheduling driven by demand and due dates
- Precedence-constrained job shop sequencing
- Routing logic across shared work centers
- Capacity limits and machine parallelism
- Bottleneck formation, utilization, and makespan tradeoffs

---

## 🎓 Who This Is For

- **Industrial Engineers** building real intuition
- **Researchers** exploring scheduling behavior
- **Students** learning production planning concepts
- **Practitioners** stress-testing assumptions

---

## ⚠️ Disclaimer

FlowLab is developed for **illustrative and educational purposes only**. It is not intended for commercial deployment or production planning in live manufacturing environments.

---

## 🛠️ Tech Stack

- Python / Streamlit
- Plotly (visualizations)
- Graphviz (routing diagrams)
- Pandas (data handling)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🤝 Contributing

Issues and PRs welcome! This is an educational project — improvements that help students and researchers understand manufacturing systems are especially valued.

---

**Define** your system → **Understand** the constraints → **Optimize** the flow

🌐 Open • 📖 Educational • 🧪 Exploratory
