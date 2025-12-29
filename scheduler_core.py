# MANUFACTURING SCHEDULER
# Clean implementation with core logic only

import math
from collections import defaultdict


# -----------------------------------------------------------------------------
# HELPERS: BOM parsing + planning (step-level inventory)
# -----------------------------------------------------------------------------

def build_bom_index(bom_rows):
    parts = {}
    for row in bom_rows:
        p = row['part_name']
        parts.setdefault(p, {'type': row['part_type'], 'steps': []})
        if row['part_type'] == 'RW':
            continue
        inputs = [s.strip() for s in row['inputs_needed'].split(',')] if row['inputs_needed'] else []
        qtys   = [int(s.strip()) for s in row['input_qty_need'].split(',')] if row['input_qty_need'] else []
        parts[p]['steps'].append({
            'stepnumber': int(row['stepnumber']),
            'workcenter': row['workcenter'],
            'batchsize':  int(row['batchsize']),
            'cycletime':  int(row['cycletime']),
            'inputs':     list(zip(inputs, qtys))
        })
    last_step = {}
    for p, rec in parts.items():
        if rec['type'] != 'RW':
            rec['steps'].sort(key=lambda s: s['stepnumber'])
            last_step[p] = rec['steps'][-1]['stepnumber']
    return parts, last_step

def plan_with_step_inventory(orders, bom_index, last_step_by_part):
    def to_int_date(d): return int((d or "99991231").replace('-', ''))
    orders_sorted = sorted(orders, key=lambda o: to_int_date(o['due_date']))

    inv_step = defaultdict(int)     # {(part, stepnumber): qty}
    raw_req  = defaultdict(int)     # {raw: total_qty}
    ledger   = []                   # actions in order

    def steps_of(part): return bom_index[part]['steps']

    def step_info(part, stepnum):
        for s in steps_of(part):
            if s['stepnumber'] == stepnum:
                return s
        raise ValueError(f"Step {stepnum} not found for {part}")

    def is_raw(part): return bom_index.get(part, {}).get('type') == 'RW'

    def ensure_output_at_step(part, stepnum, qty_needed, ctx):
        have = inv_step[(part, stepnum)]
        if have >= qty_needed:
            inv_step[(part, stepnum)] -= qty_needed
            ledger.append({
                'order': ctx['order_number'], 'due': ctx['due_date'],
                'part': part, 'step': stepnum, 'action': 'consume_step_inv',
                'qty': qty_needed, 'inv_after': inv_step[(part, stepnum)],
                'note': f'Use existing inv at ({part}, step {stepnum})'
            })
            return

        net = qty_needed - have
        inv_step[(part, stepnum)] = 0  # consumed all

        if bom_index[part]['type'] == 'RW':
            raw_req[part] += net
            ledger.append({
                'order': ctx['order_number'], 'due': ctx['due_date'],
                'part': part, 'step': None, 'action': 'raw_req',
                'qty': net, 'note': 'Raw material needed'
            })
            return

        s = step_info(part, stepnum)
        batches = math.ceil(net / s['batchsize'])
        qty_out = batches * s['batchsize']

        # Ensure BOM inputs for THIS step
        for comp, per_unit in s['inputs']:
            comp_need = qty_out * per_unit
            if is_raw(comp):
                raw_req[comp] += comp_need
                ledger.append({
                    'order': ctx['order_number'], 'due': ctx['due_date'],
                    'part': comp, 'step': None, 'action': 'raw_req',
                    'qty': comp_need, 'note': f'Raw for {part} step {stepnum}'
                })
            else:
                comp_last = last_step_by_part[comp]
                ensure_output_at_step(comp, comp_last, comp_need, ctx)

        # Ensure previous internal step exists if any
        first_step = steps_of(part)[0]['stepnumber']
        if stepnum > first_step:
            prev_step = stepnum - 1
            ensure_output_at_step(part, prev_step, qty_out, ctx)

        # Produce at this step, then consume net
        inv_step[(part, stepnum)] += qty_out
        inv_step[(part, stepnum)] -= net
        ledger.append({
            'order': ctx['order_number'], 'due': ctx['due_date'],
            'part': part, 'step': stepnum, 'action': 'produce_step',
            'batches': batches, 'batchsize': s['batchsize'], 'qty_out': qty_out,
            'consumed_now': net, 'inv_after': inv_step[(part, stepnum)],
            'workcenter': s['workcenter'],
            'note': f'Produced {qty_out} at step {stepnum} for need {qty_needed}'
        })

    for o in orders_sorted:
        p = o['product']
        if p not in bom_index or bom_index[p]['type'] == 'RW':
            raw_req[p] += o['quantity']
            ledger.append({
                'order': o['order_number'], 'due': o['due_date'],
                'part': p, 'step': None, 'action': 'raw_req',
                'qty': o['quantity'], 'note': 'Order is raw material'
            })
            continue

        last_step = last_step_by_part[p]
        ensure_output_at_step(p, last_step, o['quantity'], {'order_number': o['order_number'], 'due_date': o['due_date']})

    return {'inventory_by_step': dict(inv_step), 'raw_requirements': dict(raw_req), 'ledger': ledger}

# -----------------------------------------------------------------------------
# WORK-ORDER GENERATION (AND-gate, step bins)
# -----------------------------------------------------------------------------

def build_work_orders_from_plan_AND_gate_stepbins(plan, bom_index):
    import itertools

    def steps_of(part): return bom_index[part]['steps']
    def prev_stepnum(part, stepnum):
        steps = steps_of(part)
        idx = [i for i,s in enumerate(steps) if s['stepnumber']==stepnum][0]
        return None if idx==0 else steps[idx-1]['stepnumber']
    def last_stepnum(part): return bom_index[part]['steps'][-1]['stepnumber']

    work_orders = []
    next_id = itertools.count(1)

    for row in plan['ledger']:
        if row.get('action') != 'produce_step':
            continue

        part       = row['part']
        stepnum    = row['step']
        wc         = row['workcenter']
        batches    = row['batches']
        batchsize  = row['batchsize']
        due_date   = row['due']
        order_no   = row['order']

        step_def = next(s for s in bom_index[part]['steps'] if s['stepnumber']==stepnum)
        cycle_time = step_def['cycletime']

        inputs_template = []
        ps = prev_stepnum(part, stepnum)
        if ps is not None:
            inputs_template.append({'bin': (part, ps), 'qty_per_batch': batchsize})
        for comp, per_unit in step_def['inputs']:
            qty_per_batch = per_unit * batchsize
            ctype = bom_index.get(comp, {}).get('type')
            if ctype == 'RW':
                inputs_template.append({'bin': (comp, 'RAW'), 'qty_per_batch': qty_per_batch})
            else:
                inputs_template.append({'bin': (comp, last_stepnum(comp)), 'qty_per_batch': qty_per_batch})

        for _ in range(batches):
            work_orders.append({
                'run_id': next(next_id),
                'order': order_no,
                'due_date': due_date,
                'product': part,
                'step': stepnum,
                'process': wc,                 # queue key
                'equipment_type': wc,          # same as process here
                'inputs': inputs_template,     # AND-gate bins
                'output_bin': (part, stepnum),
                'output_qty': batchsize,
                'cycle_time': cycle_time,
                'status': 'pending'
            })

    return work_orders

# -----------------------------------------------------------------------------
# SCHEDULER (AND-gate step bins)
# -----------------------------------------------------------------------------

def initialize_inventory_state_AND_gate_stepbins(runs):
    bins = set()
    for r in runs:
        bins.add(r['output_bin'])
        for i in r['inputs']:
            bins.add(tuple(i['bin']))
    inventory = {}
    for (p, step) in bins:
        inventory[(p, step)] = 10**15 if step == 'RAW' else 0
    return inventory

def initialize_equipment_state(factory_equipment):
    equipment_state = {}
    equipment_last_product = {}
    equipment_changeover_product = {}
    for process, num_units in factory_equipment.items():
        for unit_num in range(1, num_units + 1):
            key = (process, unit_num)
            equipment_state[key] = 0
            equipment_last_product[key] = None
            equipment_changeover_product[key] = None
    return equipment_state, equipment_last_product, equipment_changeover_product

def calculate_total_time(equipment, current_product, cycle_time, changeover_time,
                         equipment_last_product, equipment_changeover_product):
    last_product = equipment_last_product.get(equipment)
    changeover_product = equipment_changeover_product.get(equipment)
    if current_product == last_product or current_product == changeover_product:
        return cycle_time, False
    else:
        equipment_changeover_product[equipment] = current_product
        return cycle_time + changeover_time, True

def schedule_single_configuration_AND_gate_stepbins(runs, factory_equipment, changeover_time=0):
    import heapq
    from collections import defaultdict

    def _key(run): return (run['due_date'], run['run_id'])

    blocked_by_process = defaultdict(list)   # waiting (missing inputs)
    ready_by_process   = defaultdict(list)   # all inputs reserved

    equipment_state, equipment_last_product, equipment_changeover_product = initialize_equipment_state(factory_equipment)
    inventory_state = initialize_inventory_state_AND_gate_stepbins(runs)
    events = []  # (time, event_type, payload)

    # Seed blocked queues with pending runs
    for r in runs:
        if r['status'] == 'pending':
            heapq.heappush(blocked_by_process[r['process']], (_key(r), r))

    def _can_and_consume_inputs(run):
        reqs = [(tuple(i['bin']), i['qty_per_batch']) for i in run['inputs']]
        if any(inventory_state.get(b, 0) < need for b, need in reqs):
            return False
        for b, need in reqs:
            inventory_state[b] -= need
        return True

    def _promote_blocked(process):
        heap = blocked_by_process[process]
        moved = 0
        while heap:
            (_, cand) = heap[0]
            if _can_and_consume_inputs(cand):
                heapq.heappop(heap)
                heapq.heappush(ready_by_process[process], (_key(cand), cand))
                moved += 1
            else:
                break
        return moved

    def _try_schedule_on_equipment(equipment, now):
        process, _ = equipment
        if not ready_by_process[process]:
            return False
        _, run = heapq.heappop(ready_by_process[process])

        total_time, needs_changeover = calculate_total_time(
            equipment, run['product'], run['cycle_time'], changeover_time,
            equipment_last_product, equipment_changeover_product
        )
        start = now
        end = now + total_time
        equipment_state[equipment] = end
        equipment_last_product[equipment] = run['product']
        if needs_changeover:
            equipment_changeover_product[equipment] = run['product']

        run['status'] = 'scheduled'
        run['start_time'] = start
        run['end_time'] = end
        run['equipment_unit'] = equipment

        out_bin = tuple(run['output_bin'])
        heapq.heappush(events, (end, 'equipment_available', equipment))
        heapq.heappush(events, (end, 'inventory_available', (out_bin, run['output_qty'])))
        return True

    # Initial promotions
    for proc in list(blocked_by_process.keys()):
        _promote_blocked(proc)

    # Try assignments at time 0
    for equipment, t in equipment_state.items():
        if t == 0:
            _try_schedule_on_equipment(equipment, 0)

    # Event loop
    while events:
        current_time, event_type, event_data = heapq.heappop(events)
        batch = [(event_type, event_data)]
        while events and events[0][0] == current_time:
            _, et, ed = heapq.heappop(events)
            batch.append((et, ed))

        # 1) apply arrivals
        for et, ed in batch:
            if et == 'inventory_available':
                bin_key, qty = ed
                inventory_state[tuple(bin_key)] = inventory_state.get(tuple(bin_key), 0) + qty

        # 2) re-promote all processes (new stock may unblock)
        for proc in list(blocked_by_process.keys()):
            _promote_blocked(proc)

        # 3) free machines at this tick
        for et, ed in batch:
            if et == 'equipment_available':
                _try_schedule_on_equipment(ed, current_time)

        # 4) scan idle machines too
        for eq, available_time in equipment_state.items():
            if available_time <= current_time:
                _try_schedule_on_equipment(eq, current_time)

    return runs


def _parse_due_date_str(d):
    """
    Accepts 'YYYY-MM-DD' or 'YYYY/MM/DD'. Returns a date object; None if bad.
    """
    if not d:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.datetime.strptime(d, fmt).date()
        except ValueError:
            continue
    return None


def _compute_workcenter_stage_map(bom_index):
    """
    Assign a 'stage score' per workcenter so we can order the Y axis:
      - SA (sub-assembly) earlier: base = 100
      - FA (final assembly) later: base = 200
      - Within part type, higher stepnumber = later
    We keep the MAX stage we ever see for a WC so FA steps push it down.
    """
    wc_stage = {}
    for _, rec in bom_index.items():
        ptype = rec.get('type')
        if ptype == 'RW':
            continue
        base = 100 if ptype == 'SA' else 200
        for s in rec['steps']:
            wc = s['workcenter']
            stage = base + int(s['stepnumber'])
            wc_stage[wc] = max(wc_stage.get(wc, 0), stage)
    return wc_stage
# -----------------------------------------------------------------------------
# ROBUST GANTT PLOT (datetime, color-by-order, FA last) + MAIN DRIVER
# -----------------------------------------------------------------------------

import datetime
from datetime import timedelta
from collections import defaultdict
import pandas as pd
import plotly.express as px
def gantt_from_scheduled_datetime_sorted(
    scheduled_runs,
    bom_index,
    title='Factory Schedule',
    base_start=None,      # None -> today 00:00
    time_units="h",       # 'h' or 'm'
    color_by='order',     # 'order' or 'product' etc.
    show_due_date_lines=True
):
    """
    Build a clean Plotly timeline from scheduled runs (numeric time),
    styled for presentation:
      - Real datetime axis (anchored at base_start)
      - Y ordered by process stage (SA → FA)
      - Minimal, clean "Apple-ish" layout
      - Optional due-date vertical markers
    """
    # 1) Build rows table from scheduled runs
    rows = []
    for r in scheduled_runs:
        if r.get('status') != 'scheduled':
            continue

        eu = r.get('equipment_unit')
        if not isinstance(eu, tuple) or len(eu) != 2:
            continue

        proc, unit = eu
        start_num = r.get('start_time', 0.0)
        finish_num = r.get('end_time', 0.0)

        if not isinstance(start_num, (int, float)) or not isinstance(finish_num, (int, float)):
            continue
        if finish_num <= start_num:
            continue

        rows.append({
            'equipment': f"{proc} #{unit}",
            'process': proc,
            'unit': unit,
            color_by: r.get(color_by),
            'product': r.get('product'),
            'order': r.get('order'),
            'step': r.get('step'),
            'run_id': r.get('run_id'),
            'due_date': r.get('due_date'),
            'start_num': float(start_num),
            'finish_num': float(finish_num),
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # 2) Anchor numeric time to real datetime
    if base_start is None:
        base_start = datetime.datetime.today().replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    if time_units.lower() == "h":
        df['start_dt']  = [base_start + timedelta(hours=t) for t in df['start_num']]
        df['finish_dt'] = [base_start + timedelta(hours=t) for t in df['finish_num']]
    elif time_units.lower() == "m":
        df['start_dt']  = [base_start + timedelta(minutes=t) for t in df['start_num']]
        df['finish_dt'] = [base_start + timedelta(minutes=t) for t in df['finish_num']]
    else:
        raise ValueError("time_units must be 'h' or 'm'.")

    # 3) Y-axis order: SA workcenters first, FA workcenters last
    wc_stage = _compute_workcenter_stage_map(bom_index)
    df['wc_stage'] = df['process'].map(lambda wc: wc_stage.get(wc, 0))
    eq_sorted = (
        df[['equipment', 'process', 'unit', 'wc_stage']]
        .drop_duplicates()
        .sort_values(['wc_stage', 'process', 'unit'])
        ['equipment']
        .tolist()
    )

    # 4) Build timeline
    df_sorted = df.sort_values(['equipment', 'start_dt'])

    fig = px.timeline(
        df_sorted,
        x_start='start_dt',
        x_end='finish_dt',
        y='equipment',
        color=color_by,
        hover_data=[
            'run_id', 'product', 'order', 'step',
            'process', 'due_date', 'start_dt', 'finish_dt'
        ],
        # No title here - we set it in layout
    )

    # Clean, minimal styling
    fig.update_yaxes(
        autorange='reversed',
        categoryorder='array',
        categoryarray=eq_sorted,
        showgrid=False,
        title_text='',  # Remove axis title
        tickfont=dict(size=10)
    )

    fig.update_xaxes(
        title_text='',  # Remove axis title
        showline=False,
        showgrid=True,
        gridcolor='rgba(255,255,255,0.1)',
        zeroline=False,
        tickfont=dict(size=10)
    )

    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        bargap=0.15,
        height=max(400, 28 * len(eq_sorted) + 100),
        margin=dict(l=100, r=20, t=40, b=60),
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            size=11,
            color='#94A3B8'
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0.0,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=10)
        ),
    )

    # Sleeker bars
    fig.update_traces(
        marker_line_width=0,
        opacity=0.9,
        hoverlabel=dict(
            bgcolor='#1E293B',
            bordercolor='#334155',
            font=dict(size=11, color='white')
        )
    )

    # Visible window: base_start → last finish + buffer
    x0 = base_start
    x1 = df['finish_dt'].max() + timedelta(minutes=30)
    if x1 <= x0:
        x1 = x0 + timedelta(hours=1)
    fig.update_xaxes(range=[x0, x1])


    # 7) Optional due-date vertical markers (no annotation bug)
    if show_due_date_lines:
        # Unique (order, due_date) combos
        due_lines = []
        for ord_id, due in (
            df[['order', 'due_date']]
            .dropna()
            .drop_duplicates()
            .itertuples(index=False)
        ):
            d = _parse_due_date_str(due)
            if not d:
                continue
            due_dt = datetime.datetime.combine(d, datetime.time(hour=17))
            if due_dt >= x0:
                due_lines.append((ord_id, due_dt))

        for ord_id, due_dt in due_lines:
            # Subtle vertical line
            fig.add_vline(
                x=due_dt,
                line_width=1,
                line_dash="dot",
                line_color="rgba(0,0,0,0.25)",
                opacity=0.6,
            )
            # Clean label floating at the top of the chart
            fig.add_annotation(
                x=due_dt,
                y=1.02,
                xref="x",
                yref="paper",
                text=f"{ord_id} due",
                showarrow=False,
                font=dict(size=10, color="rgba(0,0,0,0.7)"),
                align="center",
            )

    return fig

# -----------------------------------------------------------------------------
# MAIN: BUILD → PLAN → WOs → SCHEDULE → DIAGNOSE → PLOT
# -----------------------------------------------------------------------------

def run_manufacturing_scheduler():
    """Main function to run the manufacturing scheduler"""
    # Build indices and plan
    bom_index, last_step_by_part = build_bom_index(bom_data)
    plan = plan_with_step_inventory(customer_orders, bom_index, last_step_by_part)

    # Generate work orders
    work_orders = build_work_orders_from_plan_AND_gate_stepbins(plan, bom_index)

    # Handle missing capacity
    wo_by_proc = defaultdict(int)
    for r in work_orders:
        wo_by_proc[r['process']] += 1

    missing_caps = [p for p in sorted(wo_by_proc.keys()) if p not in work_center_capacity]
    for p in missing_caps:
        work_center_capacity[p] = 1

    # Schedule work orders
    factory_equipment = dict(work_center_capacity)
    scheduled = schedule_single_configuration_AND_gate_stepbins(
        work_orders, factory_equipment, changeover_time=0
    )

    # Generate Gantt chart
    fig = gantt_from_scheduled_datetime_sorted(
        scheduled,
        bom_index=bom_index,
        title='Manufacturing Schedule',
        base_start=None,
        time_units='m',
        color_by='order',
        show_due_date_lines=True
    )
    if fig:
        fig.show()

    return scheduled, work_orders, plan

# ----------------------------
# Default example inputs
# ----------------------------

customer_orders = [
    {'order_number': 'ORD-001', 'customer': 'Shahrukh', 'product': 'IR_FA', 'quantity': 1, 'due_date': '2025-01-15'},
]

bom_data = [
    # IR_FA - Final Assembly (1 step)
    {'part_name': 'IR_FA', 'part_type': 'FA', 'inputs_needed': 'TB201990,TC201989-1,TC202034-1', 'input_qty_need': '1,1,1', 'stepnumber': 1, 'workcenter': '811ASMLY', 'batchsize': 1, 'cycletime': 90, 'human_need': '', 'human_hours': '', 'human_need_to': ''},

    # TB201990 - Sub-assembly (3 steps)
    {'part_name': 'TB201990', 'part_type': 'SA', 'inputs_needed': 'TB201971,T201972-4300,T201972-6700', 'input_qty_need': '1,1,1', 'stepnumber': 1, 'workcenter': '763WELDM', 'batchsize': 1, 'cycletime': 30, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'TB201990', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 2, 'workcenter': '770WHLBR', 'batchsize': 1, 'cycletime': 6, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'TB201990', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 3, 'workcenter': '771HCFIN', 'batchsize': 1, 'cycletime': 30, 'human_need': '', 'human_hours': '', 'human_need_to': ''},

    # TB201971 - Sub-assembly (1 step)
    {'part_name': 'TB201971', 'part_type': 'SA', 'inputs_needed': 'TB201970', 'input_qty_need': '1', 'stepnumber': 1, 'workcenter': '764WELDM', 'batchsize': 1, 'cycletime': 9, 'human_need': '', 'human_hours': '', 'human_need_to': ''},

    # TB201970 - Sub-assembly (3 steps)
    {'part_name': 'TB201970', 'part_type': 'SA', 'inputs_needed': 'MZ1301010034', 'input_qty_need': '1', 'stepnumber': 1, 'workcenter': '763SHR16', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'TB201970', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 2, 'workcenter': '761PUNCH', 'batchsize': 1, 'cycletime': 6, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'TB201970', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 3, 'workcenter': '763PRBRK', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},

    # T201972-4300 - Sub-assembly (2 steps)
    {'part_name': 'T201972-4300', 'part_type': 'SA', 'inputs_needed': 'MZ1304010054', 'input_qty_need': '1', 'stepnumber': 1, 'workcenter': '763SHR16', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'T201972-4300', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 2, 'workcenter': '763PRBRK', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},

    # T201972-6700 - Sub-assembly (2 steps)
    {'part_name': 'T201972-6700', 'part_type': 'SA', 'inputs_needed': 'MZ1304010054', 'input_qty_need': '1', 'stepnumber': 1, 'workcenter': '763SHR16', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'T201972-6700', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 2, 'workcenter': '763PRBRK', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},

    # TC201989-1 - Sub-assembly (3 steps) - THE BIG ONE with 10 inputs
    {'part_name': 'TC201989-1', 'part_type': 'SA', 'inputs_needed': 'TA201968,TA201969,TA201967,TC201501-105,TN202444,T202275-6544,T201966-4738,T201962-6544,T201963-6431,T201965-4738', 'input_qty_need': '1,1,1,1,1,1,1,1,1,1', 'stepnumber': 1, 'workcenter': '763WELDM', 'batchsize': 1, 'cycletime': 150, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'TC201989-1', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 2, 'workcenter': '770WHLBR', 'batchsize': 1, 'cycletime': 6, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'TC201989-1', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 3, 'workcenter': '771HCFIN', 'batchsize': 1, 'cycletime': 45, 'human_need': '', 'human_hours': '', 'human_need_to': ''},

    # TA201968 - Sub-assembly (3 steps)
    {'part_name': 'TA201968', 'part_type': 'SA', 'inputs_needed': 'MZ1307010089', 'input_qty_need': '1', 'stepnumber': 1, 'workcenter': '763BDSAW', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'TA201968', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 2, 'workcenter': '763ACRO', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'TA201968', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 3, 'workcenter': '771VIKIN', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},

    # TA201969 - Sub-assembly (2 steps)
    {'part_name': 'TA201969', 'part_type': 'SA', 'inputs_needed': 'MZ1307020040', 'input_qty_need': '1', 'stepnumber': 1, 'workcenter': '763BDSAW', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'TA201969', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 2, 'workcenter': '763ACRO', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},

    # TA201967 - Sub-assembly (2 steps)
    {'part_name': 'TA201967', 'part_type': 'SA', 'inputs_needed': 'MZ1307010089', 'input_qty_need': '1', 'stepnumber': 1, 'workcenter': '763BDSAW', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'TA201967', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 2, 'workcenter': '771VIKIN', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},

    # TC201501-105 - Sub-assembly (3 steps)
    {'part_name': 'TC201501-105', 'part_type': 'SA', 'inputs_needed': 'MZ1302010028', 'input_qty_need': '1', 'stepnumber': 1, 'workcenter': '764PSMAP', 'batchsize': 1, 'cycletime': 29, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'TC201501-105', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 2, 'workcenter': '764PSMAO', 'batchsize': 1, 'cycletime': 3, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'TC201501-105', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 3, 'workcenter': '763DRLPR', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},

    # TN202444 - Sub-assembly (2 steps)
    {'part_name': 'TN202444', 'part_type': 'SA', 'inputs_needed': 'MZ1307010001', 'input_qty_need': '1', 'stepnumber': 1, 'workcenter': '763IRONW', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'TN202444', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 2, 'workcenter': '763DRLPR', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},

    # T202275-6544 - Sub-assembly (3 steps)
    {'part_name': 'T202275-6544', 'part_type': 'SA', 'inputs_needed': 'MZ1301010034', 'input_qty_need': '1', 'stepnumber': 1, 'workcenter': '763SHR16', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'T202275-6544', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 2, 'workcenter': '763IRONW', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'T202275-6544', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 3, 'workcenter': '763PRBRK', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},

    # T201966-4738 - Sub-assembly (3 steps)
    {'part_name': 'T201966-4738', 'part_type': 'SA', 'inputs_needed': 'MZ1301010034', 'input_qty_need': '1', 'stepnumber': 1, 'workcenter': '763SHR16', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'T201966-4738', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 2, 'workcenter': '761PUNCH', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'T201966-4738', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 3, 'workcenter': '763PRBRK', 'batchsize': 1, 'cycletime': 2, 'human_need': '', 'human_hours': '', 'human_need_to': ''},

    # T201962-6544 - Sub-assembly (2 steps)
    {'part_name': 'T201962-6544', 'part_type': 'SA', 'inputs_needed': 'MZ1301010034', 'input_qty_need': '1', 'stepnumber': 1, 'workcenter': '763SHR16', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'T201962-6544', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 2, 'workcenter': '763PRBRK', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},

    # T201963-6431 - Sub-assembly (2 steps)
    {'part_name': 'T201963-6431', 'part_type': 'SA', 'inputs_needed': 'MZ1301010034', 'input_qty_need': '1', 'stepnumber': 1, 'workcenter': '763SHR16', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'T201963-6431', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 2, 'workcenter': '763PRBRK', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},

    # T201965-4738 - Sub-assembly (2 steps)
    {'part_name': 'T201965-4738', 'part_type': 'SA', 'inputs_needed': 'MZ1301010034', 'input_qty_need': '1', 'stepnumber': 1, 'workcenter': '763SHR16', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'T201965-4738', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 2, 'workcenter': '763PRBRK', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},

    # TC202034-1 - Sub-assembly (1 step)
    {'part_name': 'TC202034-1', 'part_type': 'SA', 'inputs_needed': 'TB100408-5,TB100423', 'input_qty_need': '1,1', 'stepnumber': 1, 'workcenter': '761ASMLY', 'batchsize': 1, 'cycletime': 6, 'human_need': '', 'human_hours': '', 'human_need_to': ''},

    # TB100408-5 - Sub-assembly (2 steps)
    {'part_name': 'TB100408-5', 'part_type': 'SA', 'inputs_needed': 'TB100413-5,TB100416', 'input_qty_need': '1,1', 'stepnumber': 1, 'workcenter': '761HSTUD', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'TB100408-5', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 2, 'workcenter': '761SPWLD', 'batchsize': 1, 'cycletime': 2, 'human_need': '', 'human_hours': '', 'human_need_to': ''},

    # TB100413-5 - Sub-assembly (5 steps)
    {'part_name': 'TB100413-5', 'part_type': 'SA', 'inputs_needed': 'MZ1304020031', 'input_qty_need': '1', 'stepnumber': 1, 'workcenter': '761PUNCH', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'TB100413-5', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 2, 'workcenter': '761DBURR', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'TB100413-5', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 3, 'workcenter': '761FORM2', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'TB100413-5', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 4, 'workcenter': '761TWELD', 'batchsize': 1, 'cycletime': 2, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'TB100413-5', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 5, 'workcenter': '761POLSH', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},

    # TB100416 - Sub-assembly (2 steps)
    {'part_name': 'TB100416', 'part_type': 'SA', 'inputs_needed': 'MZ1304020009', 'input_qty_need': '1', 'stepnumber': 1, 'workcenter': '761PUNCH', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'TB100416', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 2, 'workcenter': '761DBURR', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},

    # TB100423 - Sub-assembly (5 steps)
    {'part_name': 'TB100423', 'part_type': 'SA', 'inputs_needed': 'MZ1304020009', 'input_qty_need': '1', 'stepnumber': 1, 'workcenter': '761PUNCH', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'TB100423', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 2, 'workcenter': '761DBURR', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'TB100423', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 3, 'workcenter': '761FORM2', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'TB100423', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 4, 'workcenter': '761TWELD', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'TB100423', 'part_type': 'SA', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': 5, 'workcenter': '761POLSH', 'batchsize': 1, 'cycletime': 1, 'human_need': '', 'human_hours': '', 'human_need_to': ''},

    # Raw Materials (8 total)
    {'part_name': 'MZ1307010089', 'part_type': 'RW', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': '', 'workcenter': '', 'batchsize': '', 'cycletime': '', 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'MZ1307020040', 'part_type': 'RW', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': '', 'workcenter': '', 'batchsize': '', 'cycletime': '', 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'MZ1302010028', 'part_type': 'RW', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': '', 'workcenter': '', 'batchsize': '', 'cycletime': '', 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'MZ1307010001', 'part_type': 'RW', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': '', 'workcenter': '', 'batchsize': '', 'cycletime': '', 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'MZ1301010034', 'part_type': 'RW', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': '', 'workcenter': '', 'batchsize': '', 'cycletime': '', 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'MZ1304010054', 'part_type': 'RW', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': '', 'workcenter': '', 'batchsize': '', 'cycletime': '', 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'MZ1304020031', 'part_type': 'RW', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': '', 'workcenter': '', 'batchsize': '', 'cycletime': '', 'human_need': '', 'human_hours': '', 'human_need_to': ''},
    {'part_name': 'MZ1304020009', 'part_type': 'RW', 'inputs_needed': '', 'input_qty_need': '', 'stepnumber': '', 'workcenter': '', 'batchsize': '', 'cycletime': '', 'human_need': '', 'human_hours': '', 'human_need_to': ''},
]

work_center_capacity = {
    '811ASMLY': 1,
    '763WELDM': 1,
    '770WHLBR': 1,
    '771HCFIN': 1,
    '764WELDM': 1,
    '763SHR16': 1,
    '761PUNCH': 1,
    '763PRBRK': 1,
    '763BDSAW': 1,
    '763ACRO': 1,
    '771VIKIN': 1,
    '764PSMAP': 1,
    '764PSMAO': 1,
    '763DRLPR': 1,
    '763IRONW': 1,
    '761ASMLY': 1,
    '761HSTUD': 1,
    '761SPWLD': 1,
    '761DBURR': 1,
    '761FORM2': 1,
    '761TWELD': 1,
    '761POLSH': 1,
}

def run_scheduler(bom_data, customer_orders, work_center_capacity, *, base_start=None, show_chart=True):
    """
    Runs planning + scheduling.

    Returns:
      scheduled_runs: list[dict] (same objects as work_orders, with schedule fields added)
      work_orders: list[dict]
      plan: dict (includes 'ledger')
      fig: plotly Figure or None
    """
    # Build indices and plan
    bom_index, last_step_by_part = build_bom_index(bom_data)
    plan = plan_with_step_inventory(customer_orders, bom_index, last_step_by_part)

    # Generate work orders
    work_orders = build_work_orders_from_plan_AND_gate_stepbins(plan, bom_index)

    # Fill missing capacities (default 1 if unspecified)
    procs = {r['process'] for r in work_orders}
    wc = dict(work_center_capacity)
    for p in sorted(procs):
        wc.setdefault(p, 1)

    # Schedule (in-place mutation of work_orders list)
    scheduled_runs = schedule_single_configuration_AND_gate_stepbins(work_orders, wc)

    fig = None
    if show_chart:
        try:
            fig = gantt_from_scheduled_datetime_sorted(
                scheduled_runs,
                bom_index=bom_index,
                title='Manufacturing Schedule',
                base_start=base_start,
                time_units='m',
                color_by='order',
                show_due_date_lines=True
            )
        except Exception:
            fig = None

    return scheduled_runs, work_orders, plan, fig
