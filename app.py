import json
import pandas as pd
import streamlit as st

from scheduler_core import (
    run_scheduler,
    bom_data as DEFAULT_BOM,
    customer_orders as DEFAULT_ORDERS,
    work_center_capacity as DEFAULT_CAPACITY,
)

st.set_page_config(page_title="Mini Manufacturing Scheduler", layout="wide")

st.title("🏭 Mini Manufacturing Scheduler")
st.caption("Enter your data in the tables below, then click Run to generate a schedule.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Controls")
    show_chart = st.checkbox("Show Gantt chart", value=True)
    run = st.button("🚀 Run Scheduler", type="primary", use_container_width=True)
    
    st.divider()
    st.markdown("**Tips:**")
    st.markdown("- Click cells to edit")
    st.markdown("- Click + to add rows")
    st.markdown("- Press Delete to remove rows")

# --- Initialize session state with defaults ---
if "orders_df" not in st.session_state:
    st.session_state.orders_df = pd.DataFrame(DEFAULT_ORDERS)

if "bom_df" not in st.session_state:
    st.session_state.bom_df = pd.DataFrame(DEFAULT_BOM).astype(str).replace("nan", "")

if "capacity_df" not in st.session_state:
    cap_list = [{"work_center": k, "num_machines": v} for k, v in DEFAULT_CAPACITY.items()]
    st.session_state.capacity_df = pd.DataFrame(cap_list)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📋 Customer Orders", "🔧 BOM / Routing", "🏭 Work Centers", "📊 Results"])

# --- Tab 1: Customer Orders ---
with tab1:
    st.subheader("Customer Orders")
    st.caption("Add your customer orders here. Click cells to edit, use + button to add rows.")
    
    orders_df = st.data_editor(
        st.session_state.orders_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "order_number": st.column_config.TextColumn("Order #"),
            "customer": st.column_config.TextColumn("Customer"),
            "product": st.column_config.TextColumn("Product"),
            "quantity": st.column_config.NumberColumn("Quantity", min_value=1),
            "due_date": st.column_config.TextColumn("Due Date (YYYY-MM-DD)"),
        },
        hide_index=True,
        key="orders_editor"
    )
    st.session_state.orders_df = orders_df

# --- Tab 2: BOM / Routing ---
with tab2:
    st.subheader("Bill of Materials & Routing")
    st.caption("Define your products, sub-assemblies, and raw materials with their manufacturing steps.")
    
    st.info("**Part Types:** FA = Final Assembly, SA = Sub-Assembly, RW = Raw Material")
    
    bom_df = st.data_editor(
        st.session_state.bom_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "part_name": st.column_config.TextColumn("Part Name"),
            "part_type": st.column_config.SelectboxColumn("Type", options=["FA", "SA", "RW"]),
            "inputs_needed": st.column_config.TextColumn("Inputs"),
            "input_qty_need": st.column_config.TextColumn("Input Qty"),
            "stepnumber": st.column_config.TextColumn("Step #"),
            "workcenter": st.column_config.TextColumn("Work Center"),
            "batchsize": st.column_config.TextColumn("Batch Size"),
            "cycletime": st.column_config.TextColumn("Cycle Time"),
            "human_need": st.column_config.TextColumn("Workers"),
            "human_hours": st.column_config.TextColumn("Hours"),
            "human_need_to": st.column_config.TextColumn("Type"),
        },
        hide_index=True,
        key="bom_editor"
    )
    st.session_state.bom_df = bom_df

# --- Tab 3: Work Center Capacity ---
with tab3:
    st.subheader("Work Center Capacity")
    st.caption("Define how many machines/stations are available at each work center.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        capacity_df = st.data_editor(
            st.session_state.capacity_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "work_center": st.column_config.TextColumn("Work Center"),
                "num_machines": st.column_config.NumberColumn("# Machines", min_value=1, max_value=100),
            },
            hide_index=True,
            key="capacity_editor"
        )
        st.session_state.capacity_df = capacity_df
    
    with col2:
        st.metric("Total Work Centers", len(capacity_df))
        total_machines = int(capacity_df["num_machines"].sum()) if not capacity_df.empty else 0
        st.metric("Total Machines", total_machines)

# --- Tab 4: Results ---
with tab4:
    scheduled = st.session_state.get("scheduled")
    work_orders = st.session_state.get("work_orders")
    plan = st.session_state.get("plan")
    fig = st.session_state.get("fig")

    if not scheduled:
        st.info("�� Click **Run Scheduler** in the sidebar to generate results.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("✅ Scheduled Runs", len(scheduled))
        c2.metric("📦 Work Orders", len(work_orders) if work_orders else 0)
        c3.metric("📒 Ledger Rows", len(plan.get("ledger", [])) if plan else 0)

        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Scheduled Runs")
        st.dataframe(scheduled, use_container_width=True, height=320)

        with st.expander("📦 Work Orders Detail"):
            st.dataframe(work_orders, use_container_width=True, height=260)

        with st.expander("📒 Planning Ledger"):
            st.dataframe(plan.get("ledger", []), use_container_width=True, height=260)
        
        st.divider()
        scheduled_df = pd.DataFrame(scheduled)
        csv = scheduled_df.to_csv(index=False)
        st.download_button(
            "📥 Download Schedule (CSV)",
            csv,
            "schedule.csv",
            "text/csv",
            use_container_width=True
        )

# --- Run Scheduler ---
if run:
    try:
        orders = st.session_state.orders_df.to_dict("records")
        
        # Ensure due_date is string
        for o in orders:
            if pd.notna(o.get("due_date")):
                o["due_date"] = str(o["due_date"])[:10]
        
        bom = st.session_state.bom_df.to_dict("records")
        
        # Clean up BOM - convert numeric strings back to int where needed
        for row in bom:
            for key in ["stepnumber", "batchsize", "cycletime"]:
                val = row.get(key, "")
                if val and val != "":
                    try:
                        row[key] = int(float(val))
                    except:
                        pass
        
        cap_df = st.session_state.capacity_df
        capacity = dict(zip(cap_df["work_center"], cap_df["num_machines"].astype(int)))

        with st.spinner("🔄 Running scheduler..."):
            scheduled, work_orders, plan, fig = run_scheduler(
                bom, orders, capacity, show_chart=show_chart
            )

        st.session_state["scheduled"] = scheduled
        st.session_state["work_orders"] = work_orders
        st.session_state["plan"] = plan
        st.session_state["fig"] = fig
        
        st.success(f"✅ Done! Generated {len(scheduled)} scheduled runs.")
        st.info("👉 Go to the **Results** tab to view the schedule and Gantt chart.")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
