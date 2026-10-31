"""
AI Water Jug Problem Solver
Beginner-level BCA AI mini-project
Algorithms: BFS, DFS, A*
Run:
    pip install streamlit
    streamlit run water_jug_solver.py
"""

from collections import deque
import heapq
import itertools
import time
import streamlit as st


# -------------------- Problem rules --------------------
def next_states(state, cap_a, cap_b):
    """Return (next_state, action_description) for each valid operation."""
    a, b = state
    results = []

    # Fill a jug
    results.append(((cap_a, b), "Fill Jug A"))
    results.append(((a, cap_b), "Fill Jug B"))

    # Empty a jug
    results.append(((0, b), "Empty Jug A"))
    results.append(((a, 0), "Empty Jug B"))

    # Pour A into B
    amount = min(a, cap_b - b)
    results.append(((a - amount, b + amount), "Pour A → B"))

    # Pour B into A
    amount = min(b, cap_a - a)
    results.append(((a + amount, b - amount), "Pour B → A"))

    # Ignore operations that do not change the state and remove duplicates
    unique = []
    seen = set()
    for new_state, action in results:
        if new_state != state and new_state not in seen:
            seen.add(new_state)
            unique.append((new_state, action))
    return unique


def is_goal(state, target):
    """Goal is reached when either jug contains the target amount."""
    return state[0] == target or state[1] == target


def rebuild(parent, end_state):
    """Create the path as (state, action-used-to-reach-it)."""
    result = []
    current = end_state
    while current is not None:
        previous, action = parent[current]
        result.append((current, action))
        current = previous
    result.reverse()
    result[0] = (result[0][0], "Start")
    return result


# -------------------- Search algorithms --------------------
def bfs(start, target, cap_a, cap_b):
    queue = deque([start])
    parent = {start: (None, None)}
    visit_order = []

    while queue:
        current = queue.popleft()
        visit_order.append(current)

        if is_goal(current, target):
            return rebuild(parent, current), visit_order

        for nxt, action in next_states(current, cap_a, cap_b):
            if nxt not in parent:
                parent[nxt] = (current, action)
                queue.append(nxt)

    return [], visit_order


def dfs(start, target, cap_a, cap_b):
    stack = [start]
    parent = {start: (None, None)}
    visit_order = []

    while stack:
        current = stack.pop()
        visit_order.append(current)

        if is_goal(current, target):
            return rebuild(parent, current), visit_order

        # Mark states when added so cycles are avoided.
        # Reverse makes the operation order easier to reproduce.
        for nxt, action in reversed(next_states(current, cap_a, cap_b)):
            if nxt not in parent:
                parent[nxt] = (current, action)
                stack.append(nxt)

    return [], visit_order


def astar(start, target, cap_a, cap_b):
    """
    A* uses f(n) = g(n) + h(n).
    h(n)=0 at a goal and 1 otherwise.
    This is admissible: every non-goal state needs at least one action.
    It is intentionally simple for teaching; it is a weak heuristic.
    """
    def heuristic(state):
        return 0 if is_goal(state, target) else 1

    counter = itertools.count()
    frontier = [(heuristic(start), next(counter), start)]
    parent = {start: (None, None)}
    g_cost = {start: 0}
    closed = set()
    visit_order = []

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current in closed:
            continue

        closed.add(current)
        visit_order.append(current)

        if is_goal(current, target):
            return rebuild(parent, current), visit_order

        for nxt, action in next_states(current, cap_a, cap_b):
            new_cost = g_cost[current] + 1
            if nxt not in g_cost or new_cost < g_cost[nxt]:
                g_cost[nxt] = new_cost
                parent[nxt] = (current, action)
                priority = new_cost + heuristic(nxt)
                heapq.heappush(frontier, (priority, next(counter), nxt))

    return [], visit_order


# -------------------- Interface --------------------
st.set_page_config(page_title="AI Water Jug Solver", page_icon="💧", layout="wide")
st.title("💧 AI Water Jug Problem Solver")
st.caption("A beginner-friendly AI project demonstrating BFS, DFS and A* search.")

with st.sidebar:
    st.header("Problem settings")
    cap_a = st.number_input("Jug A capacity (litres)", min_value=1, max_value=20, value=4, step=1)
    cap_b = st.number_input("Jug B capacity (litres)", min_value=1, max_value=20, value=3, step=1)
    target = st.number_input("Target amount (litres)", min_value=1, max_value=int(max(cap_a, cap_b)), value=min(2, int(max(cap_a, cap_b))), step=1)
    algorithm = st.selectbox("Choose search algorithm", ["BFS", "DFS", "A*"])
    st.info("Allowed actions: fill, empty, or pour between jugs.")

st.markdown(
    f"**Goal:** Measure exactly **{int(target)} L** in either jug.  \n"
    f"**Capacities:** Jug A = {int(cap_a)} L, Jug B = {int(cap_b)} L."
)

# Classic state begins with both jugs empty.
start = (0, 0)
if target > max(cap_a, cap_b):
    st.error("The target must not exceed either jug's capacity.")
else:
    if algorithm == "BFS":
        search_fn = bfs
    elif algorithm == "DFS":
        search_fn = dfs
    else:
        search_fn = astar

    start_time = time.perf_counter()
    solution, visited = search_fn(start, int(target), int(cap_a), int(cap_b))
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("States explored", len(visited))
    metric2.metric("Operations in solution", max(0, len(solution) - 1) if solution else "No solution")
    metric3.metric("Search time", f"{elapsed_ms:.3f} ms")

    if solution:
        st.success(f"{algorithm} found a solution!")
        st.subheader("Step-by-step solution")
        st.caption("State is shown as (water in Jug A, water in Jug B).")
        for index, (state, action) in enumerate(solution):
            a, b = state
            with st.container(border=True):
                col1, col2, col3 = st.columns([1, 2, 2])
                col1.markdown(f"**Step {index}**")
                col2.markdown(f"**{action}**")
                col3.markdown(f"Jug A: **{a}/{int(cap_a)} L**  \nJug B: **{b}/{int(cap_b)} L**")
                bar1, bar2 = st.columns(2)
                bar1.progress(a / cap_a, text=f"Jug A: {a} L")
                bar2.progress(b / cap_b, text=f"Jug B: {b} L")

        with st.expander("See explored states"):
            st.write(visited)
    else:
        st.warning("No solution found for these capacities and target.")

st.divider()
st.subheader("How the algorithms work")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### BFS")
    st.write("Uses a queue (first in, first out). It finds a solution with the fewest actions when each action has equal cost.")
with c2:
    st.markdown("### DFS")
    st.write("Uses a stack (last in, first out). It explores deeply first and may find a longer solution.")
with c3:
    st.markdown("### A*")
    st.write("Ranks states using f(n) = g(n) + h(n). This demo uses a safe, simple heuristic: 0 at a goal, otherwise 1.")

st.info(
    "**AI concepts:** State = (A, B); initial state = (0, 0); goal test = either jug has the target; "
    "actions = fill, empty, pour; step cost = 1."
)
st.caption("Educational simulation. The example assumes ideal jugs and no water loss.")
