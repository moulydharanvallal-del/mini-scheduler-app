import json
from datetime import datetime, timedelta
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
    '#818CF8',  # Indigo (brighter)
    '#34D399',  # Emerald (brighter)  
    '#F472B6',  # Pink (brighter)
    '#FBBF24',  # Amber (brighter)
    '#FB923C',  # Orange (brighter)
    '#22D3EE',  # Cyan (brighter)
    '#A78BFA',  # Violet (brighter)
    '#A3E635',  # Lime (brighter)
    '#F87171',  # Red (brighter)
    '#2DD4BF',  # Teal (brighter)
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

def generate_routing_graphviz(bom_df, view_mode='part'):
    """Generate modern-styled Graphviz DOT diagram from BOM data
    
    Args:
        bom_df: DataFrame with BOM data
        view_mode: 'part' for part-level view, 'step' for step-level view
    """
    
    # Create dynamic color map for this BOM
    color_map = get_workcenter_color_map(bom_df)
    
    # Part type colors (for part view)
    PART_TYPE_COLORS = {
        'FA': '#8B5CF6',  # Purple for Final Assembly
        'SA': '#3B82F6',  # Blue for Sub-Assembly
        'RW': '#475569',  # Gray for Raw Material
    }
    
    def get_modern_color(wc):
        return get_color_for_workcenter(wc, color_map)
    
    def get_part_type_color(part_type):
        return PART_TYPE_COLORS.get(part_type, '#475569')
    
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
    lines.append('    // Modern dark theme with smooth edges')
    lines.append('    bgcolor="#1E293B";')
    lines.append('    pad="0.5";')
    lines.append('    nodesep="1.0";')
    lines.append('    ranksep="1.5";')
    lines.append('    rankdir=LR;')
    lines.append('    splines=curved;')
    lines.append('    overlap=false;')
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
    lines.append('    // Elegant edge styling')
    lines.append('    edge [')
    lines.append('        color="#CBD5E1",')
    lines.append('        penwidth=1.2,')
    lines.append('        arrowsize=0.7,')
    lines.append('        arrowhead=vee,')
    lines.append('        style=solid')
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
    lines.append('        bgcolor="#334155";')
    lines.append('        color="#475569";')
    lines.append('        penwidth=2;')
    for rm in sorted(raw_materials):
        safe_id = rm.replace(' ', '_').replace('-', '_')
        lines.append(f'        {safe_id} [label=<<B>{rm}</B>>, fillcolor="#475569"];')
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
    
    if view_mode == 'step':
        # STEP VIEW: Show each step as a separate node, colored by WC
        # Sub-assemblies - all steps
        for sa in sorted(sub_assemblies):
            info = parts[sa]
            sorted_steps = sorted(info['steps'], key=lambda x: str(x['step']))
            for step_info in sorted_steps:
                step = step_info['step']
                wc = step_info['workcenter']
                color = get_modern_color(wc)
                safe_id = f"{sa}_S{step}".replace(' ', '_').replace('-', '_')
                lines.append(f'        {safe_id} [label=<<TABLE BORDER="0" CELLPADDING="2" CELLSPACING="0"><TR><TD ALIGN="LEFT"><B>{sa}</B><BR/><FONT POINT-SIZE="8">Step {step} | WC: {wc}</FONT></TD></TR></TABLE>>, fillcolor="{color}"];')
        
        # Final assemblies - all steps
        for fa in sorted(final_products):
            info = parts[fa]
            sorted_steps = sorted(info['steps'], key=lambda x: str(x['step']))
            for step_info in sorted_steps:
                step = step_info['step']
                wc = step_info['workcenter']
                color = get_modern_color(wc)
                safe_id = f"{fa}_S{step}".replace(' ', '_').replace('-', '_')
                lines.append(f'        {safe_id} [label=<<TABLE BORDER="0" CELLPADDING="2" CELLSPACING="0"><TR><TD ALIGN="LEFT"><B>{fa}</B><BR/><FONT POINT-SIZE="8">Step {step} | WC: {wc}</FONT></TD></TR></TABLE>>, fillcolor="{color}"];')
    else:
        # PART VIEW: One node per part, colored by part type
        for sa in sorted(sub_assemblies):
            info = parts[sa]
            num_steps = len(info['steps'])
            wcs = ', '.join(sorted(info['workcenters']))
            color = get_part_type_color('SA')
            safe_id = sa.replace(' ', '_').replace('-', '_')
            lines.append(f'        {safe_id} [label=<<TABLE BORDER="0" CELLPADDING="2" CELLSPACING="0"><TR><TD ALIGN="LEFT"><B>{sa}</B><BR/><FONT POINT-SIZE="8" COLOR="#E2E8F0">{num_steps} steps</FONT></TD></TR></TABLE>>, fillcolor="{color}"];')
        
        for fa in sorted(final_products):
            info = parts[fa]
            num_steps = len(info['steps'])
            color = get_part_type_color('FA')
            safe_id = f"{fa}_PART".replace(' ', '_').replace('-', '_')
            lines.append(f'        {safe_id} [label=<<TABLE BORDER="0" CELLPADDING="2" CELLSPACING="0"><TR><TD ALIGN="LEFT"><B>{fa}</B><BR/><FONT POINT-SIZE="8" COLOR="#E2E8F0">{num_steps} steps</FONT></TD></TR></TABLE>>, fillcolor="{color}"];')
    
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
        lines.append(f'        {safe_id} [label=<<TABLE BORDER="0" CELLPADDING="2" CELLSPACING="0"><TR><TD ALIGN="CENTER"><B>{fp}</B><BR/><FONT POINT-SIZE="9" COLOR="#E2E8F0">✓ Complete</FONT></TD></TR></TABLE>>, fillcolor="#22C55E", shape=box];')
    lines.append('    }')
    lines.append('')
    
    # EDGES
    if view_mode == 'step':
        # STEP VIEW EDGES
        # SA internal step flow + input edges
        for sa in sub_assemblies:
            info = parts[sa]
            sorted_steps = sorted(info['steps'], key=lambda x: str(x['step']))
            
            prev_step_id = None
            for step_info in sorted_steps:
                step = step_info['step']
                step_id = f"{sa}_S{step}".replace(' ', '_').replace('-', '_')
                
                # Internal flow between steps
                if prev_step_id:
                    lines.append(f'    {prev_step_id} -> {step_id} [color="#60A5FA", penwidth=1.5];')
                
                # Input edges (only for first step or steps that have inputs)
                for inp in step_info['inputs']:
                    inp_safe = inp.replace(' ', '_').replace('-', '_')
                    if inp in raw_materials:
                        lines.append(f'    {inp_safe} -> {step_id};')
                    elif inp in sub_assemblies:
                        # Connect to last step of input SA
                        inp_info = parts[inp]
                        inp_last_step = max(inp_info['steps'], key=lambda x: str(x['step']))['step']
                        inp_step_id = f"{inp}_S{inp_last_step}".replace(' ', '_').replace('-', '_')
                        lines.append(f'    {inp_step_id} -> {step_id};')
                
                prev_step_id = step_id
        
        # FA step flow + input edges
        for fa in final_products:
            info = parts[fa]
            sorted_steps = sorted(info['steps'], key=lambda x: str(x['step']))
            
            prev_step_id = None
            for step_info in sorted_steps:
                step = step_info['step']
                step_id = f"{fa}_S{step}".replace(' ', '_').replace('-', '_')
                
                if prev_step_id:
                    lines.append(f'    {prev_step_id} -> {step_id} [color="#60A5FA", penwidth=1.5];')
                
                for inp in step_info['inputs']:
                    inp_safe = inp.replace(' ', '_').replace('-', '_')
                    if inp in raw_materials:
                        lines.append(f'    {inp_safe} -> {step_id};')
                    elif inp in sub_assemblies:
                        inp_info = parts[inp]
                        inp_last_step = max(inp_info['steps'], key=lambda x: str(x['step']))['step']
                        inp_step_id = f"{inp}_S{inp_last_step}".replace(' ', '_').replace('-', '_')
                        lines.append(f'    {inp_step_id} -> {step_id};')
                
                prev_step_id = step_id
            
            # Connect last FA step to output
            if prev_step_id:
                out_id = f"{fa}_OUT".replace(' ', '_').replace('-', '_')
                lines.append(f'    {prev_step_id} -> {out_id} [color="#22C55E", penwidth=1.5];')
    
    else:
        # PART VIEW EDGES
        # SA inputs
        for sa in sub_assemblies:
            info = parts[sa]
            sa_id = sa.replace(' ', '_').replace('-', '_')
            for step_info in info['steps']:
                for inp in step_info['inputs']:
                    inp_id = inp.replace(' ', '_').replace('-', '_')
                    if inp in raw_materials or inp in sub_assemblies:
                        lines.append(f'    {inp_id} -> {sa_id};')
        
        # FA inputs and output
        for fa in final_products:
            info = parts[fa]
            fa_id = f"{fa}_PART".replace(' ', '_').replace('-', '_')
            
            for step_info in info['steps']:
                for inp in step_info['inputs']:
                    inp_id = inp.replace(' ', '_').replace('-', '_')
                    if inp in raw_materials or inp in sub_assemblies:
                        lines.append(f'    {inp_id} -> {fa_id};')
            
            out_id = f"{fa}_OUT".replace(' ', '_').replace('-', '_')
            lines.append(f'    {fa_id} -> {out_id} [color="#22C55E", penwidth=1.5];')
    
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
    
    # Production Start DateTime
    st.subheader("📅 Production Start")
    start_option = st.radio(
        "Start production:",
        ["Now", "Future"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if start_option == "Now":
        production_start_datetime = datetime.now()
    else:
        # Date picker
        future_date = st.date_input(
            "Date:",
            value=datetime.now().date() + timedelta(days=1),
            min_value=datetime.now().date()
        )
        # Time picker
        from datetime import time as dt_time
        future_time = st.time_input(
            "Time:",
            value=dt_time(8, 0),  # Default 8:00 AM
            step=1800  # 30-minute increments
        )
        production_start_datetime = datetime.combine(future_date, future_time)
    
    st.caption(f"📆 **{production_start_datetime.strftime('%Y-%m-%d %H:%M')}**")
    
    # Store in session state for results tab
    st.session_state['production_start_datetime'] = production_start_datetime
    
    st.divider()
    
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
    st.subheader("��️ Product Routing Map")
    st.caption("Visual representation of your BOM - how products flow from raw materials to finished goods.")
    
    # View mode toggle
    col1, col2 = st.columns([1, 3])
    with col1:
        view_mode = st.radio(
            "View Mode",
            options=["part", "step"],
            format_func=lambda x: "📦 Part View" if x == "part" else "🔧 Step View",
            horizontal=True,
            help="Part View: One node per part (simpler). Step View: One node per operation (detailed)."
        )
    
    # Generate and display the diagram
    try:
        dot_code = generate_routing_graphviz(st.session_state.bom_df, view_mode=view_mode)
        
        # Build dynamic color map and show legend
        color_map = get_workcenter_color_map(st.session_state.bom_df)
        
        if view_mode == 'step':
            st.markdown("**🎨 Work Center Colors:**")
            legend_cols = st.columns(min(len(color_map), 6)) if color_map else st.columns(1)
            
            for i, (wc, color) in enumerate(sorted(color_map.items())):
                legend_cols[i % len(legend_cols)].markdown(
                    f'<span style="background-color:{color};color:white;padding:4px 12px;border-radius:6px;font-weight:500;font-size:13px;">{wc}</span>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown("**🎨 Part Type Colors:**")
            legend_cols = st.columns(3)
            legend_cols[0].markdown(
                f'<span style="background-color:#8B5CF6;color:white;padding:4px 12px;border-radius:6px;font-weight:500;font-size:13px;">Final Assembly (FA)</span>',
                unsafe_allow_html=True
            )
            legend_cols[1].markdown(
                f'<span style="background-color:#3B82F6;color:white;padding:4px 12px;border-radius:6px;font-weight:500;font-size:13px;">Sub-Assembly (SA)</span>',
                unsafe_allow_html=True
            )
            legend_cols[2].markdown(
                f'<span style="background-color:#475569;color:white;padding:4px 12px;border-radius:6px;font-weight:500;font-size:13px;">Raw Material (RW)</span>',
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        # Render diagram
        st.graphviz_chart(dot_code, use_container_width=True)
        
        # Show raw DOT code in expander
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
        # Parse scheduled data
        scheduled_df = clean_for_display(scheduled)
        
        # =====================================================
        # CORE METRICS CALCULATION (all times in MINUTES)
        # =====================================================
        # CORE METRICS CALCULATION (all times in MINUTES)
        # =====================================================
        
        # Get capacity data (# machines per WC)
        cap_df = st.session_state.capacity_df
        wc_machines = dict(zip(cap_df["work_center"], cap_df["num_machines"].astype(int)))
        
        # Makespan (total production time)
        start_times = [r.get('start_time', 0) for r in scheduled]
        end_times = [r.get('end_time', 0) for r in scheduled]
        min_start = min(start_times) if start_times else 0
        max_end = max(end_times) if end_times else 0
        makespan_mins = max_end - min_start
        makespan_hours = makespan_mins / 60
        makespan_days = makespan_hours / 8  # Assuming 8-hour workdays
        
        # Work Center Load (in minutes)
        wc_load_mins = {}
        wc_run_count = {}
        for run in scheduled:
            eu = run.get('equipment_unit', '')
            if isinstance(eu, tuple):
                wc = eu[0]
            else:
                wc = str(eu).split()[0] if eu else 'Unknown'
            
            start = run.get('start_time', 0)
            end = run.get('end_time', 0)
            duration = end - start
            
            wc_load_mins[wc] = wc_load_mins.get(wc, 0) + duration
            wc_run_count[wc] = wc_run_count.get(wc, 0) + 1
        
        # Calculate load per machine for each WC
        wc_load_per_machine = {}
        for wc, total_load in wc_load_mins.items():
            num_machines = wc_machines.get(wc, 1)
            wc_load_per_machine[wc] = total_load / num_machines
        
        # Bottleneck = WC with highest load PER MACHINE
        bottleneck_wc = max(wc_load_per_machine, key=wc_load_per_machine.get) if wc_load_per_machine else "N/A"
        bottleneck_load_per_machine = wc_load_per_machine.get(bottleneck_wc, 0)
        bottleneck_machines = wc_machines.get(bottleneck_wc, 1)
        
        # Calculate utilization % = (WC Load) / (Makespan × # Machines)
        wc_utilization = {}
        for wc, total_load in wc_load_mins.items():
            num_machines = wc_machines.get(wc, 1)
            max_possible = makespan_mins * num_machines
            if max_possible > 0:
                wc_utilization[wc] = (total_load / max_possible) * 100
            else:
                wc_utilization[wc] = 0
        
        # Order completion tracking
        order_completion = {}
        for run in scheduled:
            order = run.get('order', '')
            order = run.get('order', '')
            end_time = run.get('end_time', 0)
            if order:
                if order not in order_completion or end_time > order_completion[order]:
                    order_completion[order] = end_time
        
        # Get due dates from orders
        order_due_dates = {}
        for _, row in st.session_state.orders_df.iterrows():
            order_num = row.get('order_number', '')
            due = row.get('due_date', '')
            if order_num and due:
                order_due_dates[order_num] = str(due)[:10]
        
        total_runs = len(scheduled)
        unique_orders = len(order_completion)
        unique_wcs = len(wc_load_mins)
        
        # =====================================================
        # 🎯 HERO METRICS - Big beautiful numbers
        # =====================================================
        st.markdown("## 🎯 Schedule Overview")
        
        hero_col1, hero_col2, hero_col3 = st.columns(3)
        
        with hero_col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 24px; border-radius: 16px; text-align: center;">
                <div style="font-size: 48px; font-weight: bold; color: white;">{makespan_hours:.1f}h</div>
                <div style="font-size: 14px; color: #E0E0E0; margin-top: 4px;">TOTAL MAKESPAN</div>
                <div style="font-size: 12px; color: #BDBDBD;">{makespan_mins:.0f} mins • {makespan_days:.1f} work days</div>
            </div>
            """, unsafe_allow_html=True)
        
        with hero_col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                        padding: 24px; border-radius: 16px; text-align: center;">
                <div style="font-size: 48px; font-weight: bold; color: white;">{total_runs}</div>
                <div style="font-size: 14px; color: #E0E0E0; margin-top: 4px;">OPERATIONS SCHEDULED</div>
                <div style="font-size: 12px; color: #BDBDBD;">{unique_orders} orders • {unique_wcs} work centers</div>
            </div>
            """, unsafe_allow_html=True)
        
        with hero_col3:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); 
                        padding: 24px; border-radius: 16px; text-align: center;">
                <div style="font-size: 36px; font-weight: bold; color: white;">{bottleneck_wc}</div>
                <div style="font-size: 14px; color: #E0E0E0; margin-top: 4px;">BOTTLENECK</div>
                <div style="font-size: 12px; color: #BDBDBD;">{bottleneck_load_per_machine:.0f} mins/machine ({bottleneck_machines} machines)</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # =====================================================
        # 📈 GANTT CHART - The main attraction
        # =====================================================
        st.markdown("## 📈 Production Schedule")
        
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Gantt chart not available. Enable 'Show Gantt chart' and re-run.")
        
        # =====================================================
        # 📊 WORK CENTER ANALYSIS
        # =====================================================
        st.markdown("## 🏭 Work Center Analysis")
        
        wc_col1, wc_col2 = st.columns([2, 1])
        
        with wc_col1:
            # Bar chart of WC utilization
            wc_df_chart = pd.DataFrame([
                {
                    'Work Center': wc, 
                    'Utilization %': wc_utilization.get(wc, 0),
                    'Machines': wc_machines.get(wc, 1)
                }
                for wc in sorted(wc_load_mins.keys(), key=lambda x: -wc_utilization.get(x, 0))
            ])
            
            if not wc_df_chart.empty:
                import plotly.express as px
                fig_wc = px.bar(
                    wc_df_chart, 
                    x='Work Center', 
                    y='Utilization %',
                    color='Utilization %',
                    color_continuous_scale='RdYlGn',
                    range_color=[0, 100]
                )
                fig_wc.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#94A3B8',
                    showlegend=False,
                    height=350,
                    yaxis=dict(range=[0, 105]),
                    margin=dict(t=20)
                )
                fig_wc.add_hline(y=80, line_dash="dash", line_color="#EF4444", 
                                annotation_text="80% target", annotation_position="right")
                st.plotly_chart(fig_wc, use_container_width=True)

        with wc_col2:
            st.markdown("### 📋 Utilization")
            wc_table_data = []
            for wc, mins in sorted(wc_load_mins.items(), key=lambda x: -wc_utilization.get(x[0], 0)):
                num_m = wc_machines.get(wc, 1)
                util_pct = wc_utilization.get(wc, 0)
                load_per_m = wc_load_per_machine.get(wc, 0)
                wc_table_data.append({
                    'WC': wc,
                    'Machines': num_m,
                    'Runs': wc_run_count.get(wc, 0),
                    'Util %': f"{util_pct:.0f}%",
                    'Mins/Machine': f"{load_per_m:.0f}"
                })
            
            if wc_table_data:
                st.dataframe(
                    pd.DataFrame(wc_table_data), 
                    use_container_width=True, 
                    hide_index=True,
                    height=300
                )

        
        # =====================================================
        # =====================================================
        # 📋 ORDER SUMMARY with On Track / Past Due
        # =====================================================
        st.markdown("## 📋 Order Completion Summary")
        
        # Get production start datetime from session state
        prod_start = st.session_state.get('production_start_datetime', datetime.now())
        
        order_data = []
        on_time_count = 0
        late_count = 0
        
        for order, end_mins in sorted(order_completion.items(), key=lambda x: x[1]):
            due_date_str = order_due_dates.get(order, None)
            hours = end_mins / 60
            
            # Calculate expected completion datetime
            # end_mins is the time from production start in minutes
            expected_end_datetime = prod_start + timedelta(minutes=end_mins)
            expected_end_date = expected_end_datetime.date()
            
            # Determine status
            if due_date_str and due_date_str != 'N/A':
                try:
                    due_date = datetime.strptime(due_date_str, '%Y-%m-%d').date()
                    days_diff = (due_date - expected_end_date).days
                    
                    if expected_end_date <= due_date:
                        status = "✅ On Track"
                        on_time_count += 1
                        days_info = f"{days_diff}d early" if days_diff > 0 else "On time"
                    else:
                        status = "🔴 Past Due"
                        late_count += 1
                        days_info = f"{abs(days_diff)}d late"
                except:
                    status = "⚪ Unknown"
                    days_info = "N/A"
            else:
                status = "⚪ No Due Date"
                days_info = "N/A"
            
            order_data.append({
                'Order': order,
                'Due Date': due_date_str if due_date_str else 'N/A',
                'Expected End': expected_end_datetime.strftime('%Y-%m-%d %H:%M'),
                'Duration': f"{hours:.1f}h",
                'Variance': days_info,
                'Status': status
            })

        
        # Show summary metrics
        if order_data:
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            with summary_col1:
                st.metric("📦 Total Orders", len(order_data))
            with summary_col2:
                st.metric("✅ On Track", on_time_count, delta=None)
            with summary_col3:
                if late_count > 0:
                    st.metric("🔴 Past Due", late_count, delta=f"-{late_count}", delta_color="inverse")
                else:
                    st.metric("🔴 Past Due", 0, delta="All on track!", delta_color="normal")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Create styled dataframe
            order_df = pd.DataFrame(order_data)
            
            # Apply color styling to status column
            def style_status(val):
                if "On Track" in str(val):
                    return 'background-color: #166534; color: white'
                elif "Past Due" in str(val):
                    return 'background-color: #991B1B; color: white'
                return ''
            
            styled_df = order_df.style.applymap(style_status, subset=['Status'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

        
        # =====================================================
        # 📑 DETAILED DATA (Collapsible)
        # =====================================================
        with st.expander("📑 Detailed Run Data", expanded=False):
            detail_tab1, detail_tab2, detail_tab3 = st.tabs(["Scheduled Runs", "Work Orders", "Planning Ledger"])
            
            with detail_tab1:
                display_cols = ['run_id', 'order', 'product', 'step', 'process', 'start_time', 'end_time', 'status']
                available_cols = [c for c in display_cols if c in scheduled_df.columns]
                if available_cols:
                    st.dataframe(scheduled_df[available_cols], use_container_width=True, height=400)
                else:
                    st.dataframe(scheduled_df, use_container_width=True, height=400)
            
            with detail_tab2:
                wo_df = clean_for_display(work_orders)
                st.dataframe(wo_df, use_container_width=True, height=400)
            
            with detail_tab3:
                ledger_df = clean_for_display(plan.get("ledger", []))
                st.dataframe(ledger_df, use_container_width=True, height=400)
        
        # =====================================================
        # 📥 DOWNLOADS
        # =====================================================
        st.markdown("---")
        st.markdown("### 📥 Export Data")
        
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        
        with dl_col1:
            csv = scheduled_df.to_csv(index=False)
            st.download_button(
                "📥 Full Schedule (CSV)",
                csv,
                "full_schedule.csv",
                "text/csv",
                use_container_width=True
            )
        
        with dl_col2:
            if order_data:
                order_csv = pd.DataFrame(order_data).to_csv(index=False)
                st.download_button(
                    "📋 Order Summary (CSV)",
                    order_csv,
                    "order_summary.csv",
                    "text/csv",
                    use_container_width=True
                )
        
        with dl_col3:
            if wc_table_data:
                wc_csv = pd.DataFrame(wc_table_data).to_csv(index=False)
                st.download_button(
                    "🏭 WC Analysis (CSV)",
                    wc_csv,
                    "workcenter_analysis.csv",
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
            # Get production start datetime
            prod_start_dt = st.session_state.get("production_start_datetime", datetime.now())
            scheduled, work_orders, plan, fig = run_scheduler(
                bom, orders, capacity, base_start=prod_start_dt, show_chart=show_chart
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
