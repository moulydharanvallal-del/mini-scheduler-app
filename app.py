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

# --- Helper to clean data for display ---
def clean_for_display(data):
    """Convert tuples and complex objects to strings for display"""
    if isinstance(data, list):
        df = pd.DataFrame(data)
        for col in df.columns:
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (tuple, list, dict)) else x)
        return df
    return data

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Controls")
    show_chart = st.checkbox("Show Gantt chart", value=True)
    run = st.button("🚀 Run Scheduler", type="primary", use_container_width=True)
    
    st.divider()
    if st.button("🔄 Reset to Sample Data"):
        for key in ["orders_df", "bom_df", "capacity_df", "scheduled", "work_orders", "plan", "fig"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# --- Initialize session state with defaults ---
if "orders_df" not in st.session_state:
    df = pd.DataFrame(DEFAULT_ORDERS)
    st.session_state.orders_df = df

if "bom_df" not in st.session_state:
    df = pd.DataFrame(DEFAULT_BOM)
    # Convert all to string to avoid type issues
    for col in df.columns:
        df[col] = df[col].astype(str).replace("nan", "").replace("<NA>", "")
    st.session_state.bom_df = df

if "capacity_df" not in st.session_state:
    cap_list = [{"work_center": k, "num_machines": v} for k, v in DEFAULT_CAPACITY.items()]
    st.session_state.capacity_df = pd.DataFrame(cap_list)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📋 Customer Orders", "🔧 BOM / Routing", "🏭 Work Centers", "📊 Results"])

# --- Tab 1: Customer Orders ---
with tab1:
    st.subheader("Customer Orders")
    st.caption("Edit the table below. Changes are saved automatically.")
    
    edited_orders = st.data_editor(
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
    # Update session state with edits
    st.session_state.orders_df = edited_orders
    
    st.caption(f"📌 {len(edited_orders)} orders loaded")

# --- Tab 2: BOM / Routing ---
with tab2:
    st.subheader("Bill of Materials & Routing")
    st.caption("Define your products, sub-assemblies, and raw materials.")
    
    st.info("**Part Types:** FA = Final Assembly, SA = Sub-Assembly, RW = Raw Material")
    
    edited_bom = st.data_editor(
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
    st.session_state.bom_df = edited_bom
    
    st.caption(f"📌 {len(edited_bom)} BOM rows loaded")

# --- Tab 3: Work Center Capacity ---
with tab3:
    st.subheader("Work Center Capacity")
    st.caption("Define how many machines/stations are available at each work center.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        edited_capacity = st.data_editor(
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
        st.session_state.capacity_df = edited_capacity
    
    with col2:
        st.metric("Total Work Centers", len(edited_capacity))
        total_machines = int(edited_capacity["num_machines"].sum()) if not edited_capacity.empty else 0
        st.metric("Total Machines", total_machines)

# --- Tab 4: Results ---
with tab4:
    scheduled = st.session_state.get("scheduled")
    work_orders = st.session_state.get("work_orders")
    plan = st.session_state.get("plan")
    fig = st.session_state.get("fig")

    if not scheduled:
        st.info("👈 Click **Run Scheduler** in the sidebar to generate results.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("✅ Scheduled Runs", len(scheduled))
        c2.metric("📦 Work Orders", len(work_orders) if work_orders else 0)
        c3.metric("📒 Ledger Rows", len(plan.get("ledger", [])) if plan else 0)

        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Scheduled Runs")
        # Clean data for display (convert tuples to strings)
        scheduled_df = clean_for_display(scheduled)
        st.dataframe(scheduled_df, use_container_width=True, height=320)

        with st.expander("📦 Work Orders Detail"):
            wo_df = clean_for_display(work_orders)
            st.dataframe(wo_df, use_container_width=True, height=260)

        with st.expander("📒 Planning Ledger"):
            ledger_df = clean_for_display(plan.get("ledger", []))
            st.dataframe(ledger_df, use_container_width=True, height=260)
        
        st.divider()
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
        # Get current data from session state
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
        st.rerun()  # Rerun to show results tab

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)  # Show full traceback for debugging
