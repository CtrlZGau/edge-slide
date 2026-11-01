"""Reference Code-as-Policies programs for the mock loop (no model): the scripted edge-slide solver
(scripts/edge_slide_scripted_primitives.py) ported to the sandboxed robot API, split over 3 turns.

It needs get_objects(), i.e. --obj (plate pose from the sim). It checks that the code-mode
plumbing (sandbox, API, persistence across turns, success stop) can solve the task; it is not a
claim about what the model will write.
"""

TURN1_HELPERS = '''
TABLE_HALF = 0.25
TIP = 0.008
SAFE_Z = 0.95
OVERHANG = 0.035

def plate():
    p = get_objects()["plate"]
    yaw = math.radians(p["rpy_deg"][2])
    hl, hw, hh = [e / 2 for e in p["extents_m"]]
    return np.array(p["position_m"]), yaw, hl, hw, hh

def extent_y(yaw, hl, hw):
    return abs(hl * math.sin(yaw)) + abs(hw * math.cos(yaw))

def overhang(side):
    pos, yaw, hl, hw, hh = plate()
    return side * pos[1] + extent_y(yaw, hl, hw) - TABLE_HALF

def lip_x(side, y_min):
    pos, yaw, hl, hw, hh = plate()
    a, b = np.meshgrid(np.linspace(-hl, hl, 61), np.linspace(-hw, hw, 41))
    c, s = math.cos(yaw), math.sin(yaw)
    x = pos[0] + c * a - s * b
    y = pos[1] + s * a + c * b
    m = side * y > y_min
    return float(x[m].mean()) if m.any() else float(pos[0])

pos, yaw, hl, hw, hh = plate()
side = 1.0 if pos[1] >= 0 else -1.0
down = [180.0, 0.0, 0.0]
print("plate", pos.round(3).tolist(), "yaw", round(math.degrees(yaw)), "push side", side)
result = {"side": side}
'''

TURN2_PUSH = '''
push_z = 0.80 + 0.005 + TIP
for k in range(4):
    if overhang(side) >= OVERHANG - 0.004:
        break
    pos, yaw, hl, hw, hh = plate()
    d = min(hl / max(abs(math.sin(yaw)), 1e-6), hw / max(abs(math.cos(yaw)), 1e-6))
    target_cy = side * (TABLE_HALF + OVERHANG - extent_y(yaw, hl, hw))
    start_y = pos[1] - side * (d + 0.025)
    end_y = target_cy - side * (d - 0.004)
    move_eef([pos[0], start_y, SAFE_Z - 0.07], down, 400)
    if k == 0:
        close_gripper()
    descend(push_z)
    y_cmd = end_y - side * 0.03
    move_eef([pos[0], y_cmd, push_z], down, 600)
    for i in range(12):
        o = overhang(side)
        if o >= OVERHANG - 0.003:
            break
        y_cmd = eef_pos()[1] + side * float(np.clip(OVERHANG - o, 0.003, 0.01))
        move_eef([eef_pos()[0], y_cmd, push_z], down, 200)
    lift(0.08)
    print("pass", k, "overhang cm", round(overhang(side) * 100, 2))
result = {"overhang_cm": round(overhang(side) * 100, 2)}
'''

TURN3_GRASP = '''
y_site = side * (TABLE_HALF + 0.012 + TIP)
gx = lip_x(side, abs(y_site))
mid_z = plate()[0][2]
g_rpy = [side * 90.0, 0.0, 0.0]
lift(max(0.0, SAFE_Z - eef_pos()[2]))
move_eef([0.05, side * 0.40, SAFE_Z], down, 400)
open_gripper()
move_eef([0.05, side * 0.40, SAFE_Z], g_rpy, 600)
move_eef([gx, side * 0.40, mid_z], g_rpy, 500)
move_eef([gx, y_site, mid_z], g_rpy, 400)
r = close_gripper()
print("closed", r)
lift(0.25)
hold(50)
'''


def scripted_code_calls() -> list[dict]:
    """Mock tool calls for code mode (MockToolClient format)."""
    return [{"name": "RUN_CODE", "args": {"code": c}} for c in (TURN1_HELPERS, TURN2_PUSH, TURN3_GRASP)]
