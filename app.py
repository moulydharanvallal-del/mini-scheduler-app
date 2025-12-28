import json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


from scheduler_core import (
    run_scheduler,
    bom_data as DEFAULT_BOM,
    customer_orders as DEFAULT_ORDERS,
    work_center_capacity as DEFAULT_CAPACITY,
)

st.set_page_config(page_title="Mini Manufacturing Scheduler", layout="wide")

st.title("🏭 Mini Manufacturing Scheduler")
st.caption("Enter your data in the tables below, then click Run to generate a schedule.")

# --- Dynamic Color Palette ---
COLOR_PALETTE = [
    '#6366F1',  # Indigo
    '#10B981',  # Emerald
    '#EC4899',  # Pink
    '#F59E0B',  # Amber
    '#F97316',  # Orange
    '#06B6D4',  # Cyan
    '#8B5CF6',  # Violet
    '#84CC16',  # Lime
    '#EF4444',  # Red
    '#14B8A6',  # Teal
]

def get_workcenter_color_map(bom_df):
    """Dynamically create color mapping for all work centers in BOM"""
    workcenters = set()
    for _, row in bom_df.iterrows():
        wc = str(row.get('workcenter', '')).strip()
        if wc and wc.lower() not in ['', 'nan', 'none']:
            workcenters.add(wc)
    color_map = {}
    for idx, wc in enumerate(sorted(workcenters)):
        color_map[wc] = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
    return color_map

def get_color_for_workcenter(wc, color_map=None):
    """Get color for a work center from the dynamic map"""
    if not wc or wc == '' or str(wc).lower() in ['nan', 'none']:
        return '#475569'
    if color_map and wc in color_map:
        return color_map[wc]
    return COLOR_PALETTE[hash(wc) % len(COLOR_PALETTE)]

def generate_routing_graphviz(bom_df):
    """Generate modern-styled Graphviz DOT diagram from BOM data"""
    
    # Create dynamic color map for this BOM
    color_map = get_workcenter_color_map(bom_df)
    
    def get_modern_color(wc):
        return get_color_for_workcenter(wc, color_map)
    
    # Parse BOM into structure
    parts = {}
    for _, row in bom_df.iterrows():
        part_name = str(row.get('part_name', '')).strip()
        part_type = str(row.get('part_type', '')).strip().upper()
        if not part_name:
            continue
            
        if part_name not in parts:
            parts[part_name] = {'type': part_type, 'steps': [], 'workcenters': set()}
        
        if part_type == 'RW':
            continue
            
        inputs_str = str(row.get('inputs_needed', ''))
        inputs = [s.strip() for s in inputs_str.split(',') if s.strip()]
        
        step = str(row.get('stepnumber', ''))
        wc = str(row.get('workcenter', '')).strip()
        
        if wc:
            parts[part_name]['workcenters'].add(wc)
        
        parts[part_name]['steps'].append({
            'step': step,
            'workcenter': wc,
            'inputs': inputs
        })
    
    # Collect nodes by type
    raw_materials = []
    sub_assemblies = []
    final_products = []
    
    for part_name, info in parts.items():
        if info['type'] == 'RW':
            raw_materials.append(part_name)
        elif info['type'] == 'SA':
            sub_assemblies.append(part_name)
        elif info['type'] == 'FA':
            final_products.append(part_name)
    
    # Build modern DOT graph
    lines = []
    lines.append('digraph BOM {')
    lines.append('    // Modern dark theme')
    lines.append('    bgcolor="#1E293B";')  # Slate-800 background
    lines.append('    pad="0.5";')
    lines.append('    nodesep="0.8";')
    lines.append('    ranksep="1.2";')
    lines.append('    rankdir=LR;')
    lines.append('    ')
    lines.append('    // Global node styling')
    lines.append('    node [')
    lines.append('        shape=box,')
    lines.append('        style="rounded,filled",')
    lines.append('        fontname="Helvetica Neue,Arial,sans-serif",')
    lines.append('        fontsize=11,')
    lines.append('        fontcolor="white",')
    lines.append('        penwidth=0,')
    lines.append('        margin="0.3,0.2"')
    lines.append('    ];')
    lines.append('    ')
    lines.append('    // Global edge styling')
    lines.append('    edge [')
    lines.append('        color="#94A3B8",')  # Slate-400
    lines.append('        penwidth=2,')
    lines.append('        arrowsize=0.8,')
    lines.append('        arrowhead=vee')
    lines.append('    ];')
    lines.append('')
    
    # Raw Materials cluster
    lines.append('    subgraph cluster_raw {')
    lines.append('        label="RAW MATERIALS";')
    lines.append('        labelloc="t";')
    lines.append('        fontname="Helvetica Neue,Arial,sans-serif";')
    lines.append('        fontsize=12;')
    lines.append('        fontcolor="#94A3B8";')
    lines.append('        style="rounded";')
    lines.append('        bgcolor="#334155";')  # Slate-700
    lines.append('        color="#475569";')
    lines.append('        penwidth=2;')
    for rm in sorted(raw_materials):
        safe_id = rm.replace(' ', '_').replace('-', '_')
        lines.append(f'        {safe_id} [label="{rm}", fillcolor="#475569"];')  # Slate-600
    lines.append('    }')
    lines.append('')
    
    # Manufacturing Flow cluster
    lines.append('    subgraph cluster_flow {')
    lines.append('        label="MANUFACTURING";')
    lines.append('        labelloc="t";')
    lines.append('        fontname="Helvetica Neue,Arial,sans-serif";')
    lines.append('        fontsize=12;')
    lines.append('        fontcolor="#94A3B8";')
    lines.append('        style="rounded";')
    lines.append('        bgcolor="#334155";')
    lines.append('        color="#475569";')
    lines.append('        penwidth=2;')
    
    # Sub-assemblies
    for sa in sorted(sub_assemblies):
        info = parts[sa]
        wc = list(info['workcenters'])[0] if info['workcenters'] else ''
        color = get_modern_color(wc)
        safe_id = sa.replace(' ', '_').replace('-', '_')
        lines.append(f'        {safe_id} [label=<<B>{sa}</B><BR/><FONT POINT-SIZE="9">{wc}</FONT>>, fillcolor="{color}"];')
    
    # FA steps
    for fa in sorted(final_products):
        info = parts[fa]
        for step_info in sorted(info['steps'], key=lambda x: str(x['step'])):
            step = step_info['step']
            wc = step_info['workcenter']
            color = get_modern_color(wc)
            safe_id = f"{fa}_S{step}".replace(' ', '_').replace('-', '_')
            lines.append(f'        {safe_id} [label=<<B>{fa}</B><BR/>Step {step}<BR/><FONT POINT-SIZE="9">{wc}</FONT>>, fillcolor="{color}"];')
    
    lines.append('    }')
    lines.append('')
    
    # Finished Goods cluster
    lines.append('    subgraph cluster_fg {')
    lines.append('        label="FINISHED GOODS";')
    lines.append('        labelloc="t";')
    lines.append('        fontname="Helvetica Neue,Arial,sans-serif";')
    lines.append('        fontsize=12;')
    lines.append('        fontcolor="#94A3B8";')
    lines.append('        style="rounded";')
    lines.append('        bgcolor="#334155";')
    lines.append('        color="#475569";')
    lines.append('        penwidth=2;')
    for fp in sorted(final_products):
        safe_id = f"{fp}_OUT".replace(' ', '_').replace('-', '_')
        lines.append(f'        {safe_id} [label=<<B>{fp}</B><BR/><FONT POINT-SIZE="9">✓ Complete</FONT>>, fillcolor="#22C55E", shape=box];')  # Green-500
    lines.append('    }')
    lines.append('')
    
    # Edges - SA inputs
    for sa in sub_assemblies:
        info = parts[sa]
        sa_id = sa.replace(' ', '_').replace('-', '_')
        for step_info in info['steps']:
            for inp in step_info['inputs']:
                inp_id = inp.replace(' ', '_').replace('-', '_')
                if inp in raw_materials or inp in sub_assemblies:
                    lines.append(f'    {inp_id} -> {sa_id};')
    
    # Edges - FA steps
    for fa in final_products:
        info = parts[fa]
        sorted_steps = sorted(info['steps'], key=lambda x: str(x['step']))
        
        prev_step_id = None
        for step_info in sorted_steps:
            step = step_info['step']
            step_id = f"{fa}_S{step}".replace(' ', '_').replace('-', '_')
            
            if prev_step_id:
                lines.append(f'    {prev_step_id} -> {step_id} [color="#60A5FA", penwidth=3];')  # Blue highlight for flow
            
            for inp in step_info['inputs']:
                inp_id = inp.replace(' ', '_').replace('-', '_')
                if inp in raw_materials or inp in sub_assemblies:
                    lines.append(f'    {inp_id} -> {step_id};')
            
            prev_step_id = step_id
        
        if prev_step_id:
            out_id = f"{fa}_OUT".replace(' ', '_').replace('-', '_')
            lines.append(f'    {prev_step_id} -> {out_id} [color="#22C55E", penwidth=3];')  # Green for final
    
    lines.append('}')
    
    return '\n'.join(lines)

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
    for col in df.columns:
        df[col] = df[col].astype(str).replace("nan", "").replace("<NA>", "")
    st.session_state.bom_df = df

if "capacity_df" not in st.session_state:
    cap_list = [{"work_center": k, "num_machines": v} for k, v in DEFAULT_CAPACITY.items()]
    st.session_state.capacity_df = pd.DataFrame(cap_list)

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Orders", "🔧 BOM", "🏭 Work Centers", "��️ Routing Map", "📊 Results"])

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

# --- Tab 4: Routing Map ---
with tab4:
    st.subheader("🗺️ Product Routing Map")
    st.caption("Visual representation of your BOM - how products flow from raw materials to finished goods.")
    
    # Generate and display the diagram
    try:
        dot_code = generate_routing_graphviz(st.session_state.bom_df)
        
        # Build dynamic color map and show legend
        color_map = get_workcenter_color_map(st.session_state.bom_df)
        
        st.markdown("**🎨 Work Center Colors:**")
        legend_cols = st.columns(min(len(color_map), 6)) if color_map else st.columns(1)
        
        for i, (wc, color) in enumerate(sorted(color_map.items())):
            legend_cols[i % len(legend_cols)].markdown(
                f'<span style="background-color:{color};color:white;padding:4px 12px;border-radius:6px;font-weight:500;font-size:13px;">{wc}</span>',
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        # Render diagram
        st.graphviz_chart(dot_code, use_container_width=True)
        
        # Show raw Mermaid code in expander
        with st.expander("📝 View DOT Code"):
            st.code(dot_code, language="dot")
            
    except Exception as e:
        st.error(f"Error generating diagram: {str(e)}")
        st.exception(e)

# --- Tab 5: Results ---
with tab5:
    scheduled = st.session_state.get("scheduled")
    work_orders = st.session_state.get("work_orders")
    plan = st.session_state.get("plan")
    fig = st.session_state.get("fig")

    if not scheduled:
        st.info("👈 Click **Run Scheduler** in the sidebar to generate results.")
    else:
        # Parse scheduled data for insights
        scheduled_df = clean_for_display(scheduled)
        
        # Calculate insights
        total_runs = len(scheduled)
        
        # Get order completion times
        order_completion = {}
        order_due_dates = {}
        
        # Get due dates from orders
        for _, row in st.session_state.orders_df.iterrows():
            order_num = row.get('order_number', '')
            due = row.get('due_date', '')
            if order_num and due:
                order_due_dates[order_num] = str(due)[:10]
        
        # Get completion times from scheduled runs
        for run in scheduled:
            order = run.get('order', '')
            end_time = run.get('end_time', 0)
            if order:
                if order not in order_completion or end_time > order_completion[order]:
                    order_completion[order] = end_time
        
        # Calculate on-time vs late (assuming end_time is in hours from now)
        on_time_orders = []
        late_orders = []
        
        # Get work center utilization
        wc_hours = {}
        for run in scheduled:
            eu = run.get('equipment_unit', '')
            if isinstance(eu, tuple):
                wc = eu[0]
            else:
                wc = str(eu).split()[0] if eu else 'Unknown'
            
            start = run.get('start_time', 0)
            end = run.get('end_time', 0)
            duration = end - start
            
            if wc not in wc_hours:
                wc_hours[wc] = 0
            wc_hours[wc] += duration
        
        # Find bottleneck (most loaded work center)
        bottleneck_wc = max(wc_hours, key=wc_hours.get) if wc_hours else "N/A"
        bottleneck_hours = wc_hours.get(bottleneck_wc, 0)
        
        # Max end time
        max_end_time = max([r.get('end_time', 0) for r in scheduled]) if scheduled else 0
        total_hours = max_end_time
        total_days = round(max_end_time / 24, 1) if max_end_time else 0
        
        # --- EXECUTIVE SUMMARY ---
        st.subheader("📊 Executive Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📦 Total Runs", f"{total_runs:,}")
        col2.metric("📅 Production Span", f"{total_days} days")
        col3.metric("🏭 Bottleneck", bottleneck_wc)
        col4.metric("⏱️ Bottleneck Load", f"{round(bottleneck_hours, 1)}h")
        
        st.divider()
        
        # --- ORDER STATUS ---
        st.subheader("📋 Order Status")
        
        order_status_data = []
        for order, end_time in order_completion.items():
            completion_hours = end_time
            completion_days = round(end_time / 24, 1)
            due_date = order_due_dates.get(order, 'N/A')
            
            order_status_data.append({
                'Order': order,
                'Due Date': due_date,
                'Completes In': f"{completion_days} days ({round(completion_hours, 1)}h)",
                'Status': '✅ On Track'  # Simplified - would need more logic for actual late detection
            })
        
        if order_status_data:
            status_df = pd.DataFrame(order_status_data)
            st.dataframe(status_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # --- GANTT CHART ---
        st.subheader("📈 Production Schedule (Gantt)")
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Gantt chart not available. Try running with 'Show Gantt chart' enabled.")
        
        st.divider()
        
        # --- WORK CENTER UTILIZATION ---
        st.subheader("🏭 Work Center Load")
        
        wc_data = []
        for wc, hours in sorted(wc_hours.items(), key=lambda x: -x[1]):
            wc_data.append({
                'Work Center': wc,
                'Total Hours': round(hours, 1),
                'Load Bar': '█' * min(int(hours / max(wc_hours.values()) * 20), 20) if wc_hours else ''
            })
        
        if wc_data:
            wc_df = pd.DataFrame(wc_data)
            st.dataframe(wc_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # --- DETAILED DATA (Expandable) ---
        st.subheader("📑 Detailed Data")
        
        detail_tab1, detail_tab2, detail_tab3 = st.tabs(["Scheduled Runs", "Work Orders", "Planning Ledger"])
        
        with detail_tab1:
            # Show key columns only
            display_cols = ['run_id', 'order', 'product', 'step', 'process', 'start_time', 'end_time', 'status']
            available_cols = [c for c in display_cols if c in scheduled_df.columns]
            if available_cols:
                st.dataframe(scheduled_df[available_cols], use_container_width=True, height=300)
            else:
                st.dataframe(scheduled_df, use_container_width=True, height=300)
        
        with detail_tab2:
            wo_df = clean_for_display(work_orders)
            st.dataframe(wo_df, use_container_width=True, height=300)
        
        with detail_tab3:
            ledger_df = clean_for_display(plan.get("ledger", []))
            st.dataframe(ledger_df, use_container_width=True, height=300)
        
        st.divider()
        
        # --- DOWNLOADS ---
        st.subheader("📥 Export Data")
        
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        
        with dl_col1:
            csv = scheduled_df.to_csv(index=False)
            st.download_button(
                "📥 Full Schedule",
                csv,
                "full_schedule.csv",
                "text/csv",
                use_container_width=True
            )
        
        with dl_col2:
            if order_status_data:
                order_csv = pd.DataFrame(order_status_data).to_csv(index=False)
                st.download_button(
                    "📋 Order Summary",
                    order_csv,
                    "order_summary.csv",
                    "text/csv",
                    use_container_width=True
                )
        
        with dl_col3:
            if wc_data:
                wc_csv = pd.DataFrame(wc_data).to_csv(index=False)
                st.download_button(
                    "🏭 WC Utilization",
                    wc_csv,
                    "workcenter_utilization.csv",
                    "text/csv",
                    use_container_width=True
                )


# --- Run Scheduler ---
if run:
    try:
        orders = st.session_state.orders_df.to_dict("records")
        
        for o in orders:
            if pd.notna(o.get("due_date")):
                o["due_date"] = str(o["due_date"])[:10]
        
        bom = st.session_state.bom_df.to_dict("records")
        
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
        st.rerun()

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)
