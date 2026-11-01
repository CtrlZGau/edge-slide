# Stage 3 primitive report (edge-slide, sim, no model)

**138/138 checks passed** over seeds [0, 1, 2].

Commanded vs achieved error over passing MOVE_EEF-based moves (n=84): position mean **0.27 cm**, worst **0.48 cm**; orientation mean **0.45 deg**, worst **1.05 deg**. Paper: 0.14 cm / 0.88 deg mean, 0.3 cm / 1.27 deg worst.

Gains: KP_POS=1.0, KD_POS=0.08s, KP_ROT=1.0, KD_ROT=0.1s, ROT_CAP=0.3; exit at 5 mm / 5 deg and speed < 0.02 m/s, 5.0 deg/s for 3 ticks. RPY = extrinsic XYZ (deg).

Frames: 2x2 sheet, top = before, bottom = after; left = agentview (exo_right_cam), right = gripper_cam.

| seed | group | test | target | achieved | error | steps | pass | notes | frames |
|---|---|---|---|---|---|---|---|---|---|
| 0 | position | axis x +0.05 m | [0.114, 0.025, 1.020] | [0.113, 0.025, 1.020] | 2.11 mm | 58 | PASS | disp [0.048, -0.000, 0.001] sign_ok=True ori_drift=0.24deg | [img](stage3_primitive_report/frames/s0_axis_x+0.05.jpg) |
| 0 | position | axis x -0.05 m | [0.016, 0.025, 1.019] | [0.018, 0.025, 1.018] | 2.32 mm | 57 | PASS | disp [-0.048, 0.000, -0.001] sign_ok=True ori_drift=0.19deg | [img](stage3_primitive_report/frames/s0_axis_x-0.05.jpg) |
| 0 | position | axis x +0.10 m | [0.163, 0.025, 1.020] | [0.162, 0.025, 1.022] | 2.33 mm | 72 | PASS | disp [0.099, -0.000, 0.002] sign_ok=True ori_drift=0.51deg | [img](stage3_primitive_report/frames/s0_axis_x+0.10.jpg) |
| 0 | position | axis x -0.10 m | [-0.035, 0.025, 1.018] | [-0.033, 0.025, 1.016] | 2.21 mm | 72 | PASS | disp [-0.098, -0.000, -0.001] sign_ok=True ori_drift=0.35deg | [img](stage3_primitive_report/frames/s0_axis_x-0.10.jpg) |
| 0 | position | axis y +0.05 m | [0.063, 0.075, 1.021] | [0.062, 0.073, 1.021] | 2.09 mm | 57 | PASS | disp [-0.001, 0.048, -0.000] sign_ok=True ori_drift=0.18deg | [img](stage3_primitive_report/frames/s0_axis_y+0.05.jpg) |
| 0 | position | axis y -0.05 m | [0.064, -0.024, 1.019] | [0.063, -0.022, 1.019] | 2.06 mm | 57 | PASS | disp [-0.001, -0.048, -0.001] sign_ok=True ori_drift=0.18deg | [img](stage3_primitive_report/frames/s0_axis_y-0.05.jpg) |
| 0 | position | axis y +0.10 m | [0.064, 0.124, 1.019] | [0.061, 0.123, 1.018] | 3.52 mm | 70 | PASS | disp [-0.003, 0.098, -0.002] sign_ok=True ori_drift=0.43deg | [img](stage3_primitive_report/frames/s0_axis_y+0.10.jpg) |
| 0 | position | axis y -0.10 m | [0.064, -0.074, 1.019] | [0.061, -0.073, 1.017] | 3.96 mm | 71 | PASS | disp [-0.003, -0.098, -0.002] sign_ok=True ori_drift=0.45deg | [img](stage3_primitive_report/frames/s0_axis_y-0.10.jpg) |
| 0 | position | axis z +0.05 m | [0.064, 0.025, 1.070] | [0.065, 0.025, 1.067] | 2.85 mm | 58 | PASS | disp [0.000, -0.000, 0.047] sign_ok=True ori_drift=0.10deg | [img](stage3_primitive_report/frames/s0_axis_z+0.05.jpg) |
| 0 | position | axis z -0.05 m | [0.063, 0.025, 0.972] | [0.062, 0.025, 0.974] | 2.76 mm | 57 | PASS | disp [-0.001, 0.000, -0.048] sign_ok=True ori_drift=0.21deg | [img](stage3_primitive_report/frames/s0_axis_z-0.05.jpg) |
| 0 | position | axis z +0.10 m | [0.063, 0.025, 1.118] | [0.064, 0.025, 1.115] | 2.77 mm | 74 | PASS | disp [0.001, -0.000, 0.097] sign_ok=True ori_drift=0.19deg | [img](stage3_primitive_report/frames/s0_axis_z+0.10.jpg) |
| 0 | position | axis z -0.10 m | [0.063, 0.025, 0.922] | [0.060, 0.025, 0.924] | 3.58 mm | 71 | PASS | disp [-0.003, -0.000, -0.098] sign_ok=True ori_drift=0.51deg | [img](stage3_primitive_report/frames/s0_axis_z-0.10.jpg) |
| 0 | position | random target 0 | [0.040, -0.089, 0.993] | [0.037, -0.089, 0.991] | 4.50 mm | 140 | PASS |  | [img](stage3_primitive_report/frames/s0_random_0.jpg) |
| 0 | position | random target 1 | [0.146, -0.198, 0.927] | [0.143, -0.198, 0.926] | 2.81 mm | 129 | PASS |  | [img](stage3_primitive_report/frames/s0_random_1.jpg) |
| 0 | position | random target 2 | [0.085, 0.130, 0.930] | [0.083, 0.131, 0.929] | 1.91 mm | 135 | PASS |  | [img](stage3_primitive_report/frames/s0_random_2.jpg) |
| 0 | position | random target 3 | [0.051, 0.157, 0.946] | [0.052, 0.156, 0.946] | 1.03 mm | 172 | PASS |  | [img](stage3_primitive_report/frames/s0_random_3.jpg) |
| 0 | position | random target 4 | [-0.085, -0.157, 0.948] | [-0.081, -0.158, 0.951] | 4.64 mm | 364 | PASS |  | [img](stage3_primitive_report/frames/s0_random_4.jpg) |
| 0 | position | target 5 cm below table | [0.064, 0.200, 0.750] | [0.072, 0.199, 0.809] | 59.1 mm | 200 | PASS | converged=False contact=True peakF=675.47N finite=True | [img](stage3_primitive_report/frames/s0_below_table.jpg) |
| 0 | orientation | yaw +45 deg | [-178.8, -8.2, 44.9] | [-178.9, -8.4, 43.9] | 1.04 deg | 134 | PASS | pos drift 1.67 mm; x-axis CCW from above=True; readback dyaw=+44.0deg | [img](stage3_primitive_report/frames/s0_yaw+45.jpg) |
| 0 | orientation | yaw -45 deg | [-178.8, -8.2, -45.1] | [-178.8, -8.4, -44.1] | 1.02 deg | 137 | PASS | pos drift 2.11 mm | [img](stage3_primitive_report/frames/s0_yaw-45.jpg) |
| 0 | orientation | yaw +90 deg | [-178.8, -8.2, 89.9] | [-178.8, -8.3, 88.9] | 1.02 deg | 236 | PASS | pos drift 0.32 mm; x-axis CCW from above=True; readback dyaw=+89.0deg | [img](stage3_primitive_report/frames/s0_yaw+90.jpg) |
| 0 | orientation | yaw -90 deg | [-178.8, -8.2, -90.1] | [-178.9, -8.3, -89.1] | 1.03 deg | 236 | PASS | pos drift 0.37 mm | [img](stage3_primitive_report/frames/s0_yaw-90.jpg) |
| 0 | orientation | roll +20 deg | [-158.8, -8.2, -0.1] | [-158.9, -8.0, -0.0] | 0.28 deg | 140 | PASS | pos drift 2.99 mm | [img](stage3_primitive_report/frames/s0_roll+20.jpg) |
| 0 | orientation | roll -20 deg | [-198.8, -8.2, -0.1] | [161.2, -8.0, -0.1] | 0.26 deg | 134 | PASS | pos drift 2.76 mm | [img](stage3_primitive_report/frames/s0_roll-20.jpg) |
| 0 | orientation | pitch +20 deg | [-178.8, 11.8, -0.1] | [-178.8, 11.3, -0.0] | 0.47 deg | 80 | PASS | pos drift 4.38 mm | [img](stage3_primitive_report/frames/s0_pitch+20.jpg) |
| 0 | orientation | pitch -20 deg | [-178.8, -28.2, -0.1] | [-178.8, -28.3, -0.1] | 0.08 deg | 95 | PASS | pos drift 4.75 mm | [img](stage3_primitive_report/frames/s0_pitch-20.jpg) |
| 0 | orientation | combined pos+ori | [0.124, -0.055, 0.950] [-178.8, 1.8, 34.9] | [0.125, -0.055, 0.950] [-178.7, 1.6, 33.9] | 1.33 mm / 1.03 deg | 115 | PASS |  | [img](stage3_primitive_report/frames/s0_combined.jpg) |
| 0 | orientation | round trip (readback->command) | [0.125, -0.055, 0.950] [-178.7, 1.6, 33.9] | [0.125, -0.055, 0.950] [-178.7, 1.7, 33.9] | 0.05 mm / 0.06 deg | 2 | PASS | expect ~zero motion |  |
| 0 | gripper | CLOSE (empty) | width<0.015 | 0.0099 | - | 34 | PASS | object_between=False |  |
| 0 | gripper | MOVE_EEF keeps CLOSED | 0.0099 | 0.0100 | 0.08 mm | 71 | PASS | register=1.0 | [img](stage3_primitive_report/frames/s0_closed_move.jpg) |
| 0 | gripper | OPEN | width>0.07 | 0.0771 | - | 45 | PASS |  |  |
| 0 | lift/desc | DESCEND to z=0.88 | z=0.880 | z=0.882 | xy drift 3.56 mm | 82 | PASS |  |  |
| 0 | lift/desc | LIFT +0.12 m | z=1.002 | z=1.000 | xy drift 1.97 mm | 80 | PASS |  |  |
| 0 | lift/desc | HOLD 50 ticks | drift<2mm | 0.25 mm | 0.25 mm | 50 | PASS |  |  |
| 0 | atomic | ATOMIC dx +1x0.5 x20 | sign +1 on dx | disp [0.028, -0.000, -0.002] m | - | 20 | PASS |  |  |
| 0 | atomic | ATOMIC dx -1x0.5 x20 | sign -1 on dx | disp [-0.028, -0.000, 0.002] m | - | 20 | PASS |  |  |
| 0 | atomic | ATOMIC dy +1x0.5 x20 | sign +1 on dy | disp [-0.001, 0.027, -0.000] m | - | 20 | PASS |  |  |
| 0 | atomic | ATOMIC dy -1x0.5 x20 | sign -1 on dy | disp [-0.000, -0.027, 0.000] m | - | 20 | PASS |  |  |
| 0 | atomic | ATOMIC dz +1x0.5 x20 | sign +1 on dz | disp [-0.003, -0.000, 0.029] m | - | 20 | PASS |  |  |
| 0 | atomic | ATOMIC dz -1x0.5 x20 | sign -1 on dz | disp [0.002, 0.000, -0.030] m | - | 20 | PASS |  |  |
| 0 | atomic | ATOMIC droll +1x0.5 x20 | sign +1 on droll | world rotvec [16.30, 0.26, -2.19] deg | - | 20 | PASS |  |  |
| 0 | atomic | ATOMIC droll -1x0.5 x20 | sign -1 on droll | world rotvec [-15.92, -1.33, 2.71] deg | - | 20 | PASS |  |  |
| 0 | atomic | ATOMIC dpitch +1x0.5 x20 | sign +1 on dpitch | world rotvec [0.18, 12.11, -0.17] deg | - | 20 | PASS |  |  |
| 0 | atomic | ATOMIC dpitch -1x0.5 x20 | sign -1 on dpitch | world rotvec [-0.18, -9.21, 0.25] deg | - | 20 | PASS |  |  |
| 0 | atomic | ATOMIC dyaw +1x0.5 x20 | sign +1 on dyaw | world rotvec [0.25, 0.03, 11.61] deg | - | 20 | PASS |  |  |
| 0 | atomic | ATOMIC dyaw -1x0.5 x20 | sign -1 on dyaw | world rotvec [-0.20, -0.22, -11.60] deg | - | 20 | PASS |  |  |
| 1 | position | axis x +0.05 m | [0.080, 0.009, 1.005] | [0.078, 0.009, 1.006] | 2.00 mm | 58 | PASS | disp [0.048, 0.000, 0.001] sign_ok=True ori_drift=0.25deg | [img](stage3_primitive_report/frames/s1_axis_x+0.05.jpg) |
| 1 | position | axis x -0.05 m | [-0.019, 0.009, 1.004] | [-0.017, 0.008, 1.003] | 2.20 mm | 57 | PASS | disp [-0.048, -0.000, -0.001] sign_ok=True ori_drift=0.21deg | [img](stage3_primitive_report/frames/s1_axis_x-0.05.jpg) |
| 1 | position | axis x +0.10 m | [0.128, 0.009, 1.006] | [0.127, 0.009, 1.008] | 2.26 mm | 71 | PASS | disp [0.099, 0.000, 0.002] sign_ok=True ori_drift=0.53deg | [img](stage3_primitive_report/frames/s1_axis_x+0.10.jpg) |
| 1 | position | axis x -0.10 m | [-0.069, 0.008, 1.003] | [-0.068, 0.008, 1.002] | 2.18 mm | 71 | PASS | disp [-0.098, -0.000, -0.001] sign_ok=True ori_drift=0.38deg | [img](stage3_primitive_report/frames/s1_axis_x-0.10.jpg) |
| 1 | position | axis y +0.05 m | [0.029, 0.059, 1.006] | [0.028, 0.056, 1.006] | 2.89 mm | 55 | PASS | disp [-0.001, 0.047, -0.000] sign_ok=True ori_drift=0.14deg | [img](stage3_primitive_report/frames/s1_axis_y+0.05.jpg) |
| 1 | position | axis y -0.05 m | [0.029, -0.039, 1.005] | [0.029, -0.036, 1.005] | 2.81 mm | 55 | PASS | disp [-0.001, -0.047, -0.000] sign_ok=True ori_drift=0.14deg | [img](stage3_primitive_report/frames/s1_axis_y-0.05.jpg) |
| 1 | position | axis y +0.10 m | [0.029, 0.106, 1.005] | [0.028, 0.104, 1.004] | 3.17 mm | 70 | PASS | disp [-0.002, 0.097, -0.001] sign_ok=True ori_drift=0.28deg | [img](stage3_primitive_report/frames/s1_axis_y+0.10.jpg) |
| 1 | position | axis y -0.10 m | [0.029, -0.089, 1.005] | [0.027, -0.087, 1.003] | 3.44 mm | 70 | PASS | disp [-0.002, -0.097, -0.001] sign_ok=True ori_drift=0.31deg | [img](stage3_primitive_report/frames/s1_axis_y-0.10.jpg) |
| 1 | position | axis z +0.05 m | [0.029, 0.007, 1.055] | [0.030, 0.007, 1.052] | 2.94 mm | 58 | PASS | disp [0.000, -0.000, 0.047] sign_ok=True ori_drift=0.09deg | [img](stage3_primitive_report/frames/s1_axis_z+0.05.jpg) |
| 1 | position | axis z -0.05 m | [0.029, 0.009, 0.957] | [0.027, 0.009, 0.960] | 2.84 mm | 57 | PASS | disp [-0.001, 0.000, -0.047] sign_ok=True ori_drift=0.22deg | [img](stage3_primitive_report/frames/s1_axis_z-0.05.jpg) |
| 1 | position | axis z +0.10 m | [0.028, 0.009, 1.103] | [0.029, 0.008, 1.100] | 2.96 mm | 74 | PASS | disp [0.001, -0.000, 0.097] sign_ok=True ori_drift=0.14deg | [img](stage3_primitive_report/frames/s1_axis_z+0.10.jpg) |
| 1 | position | axis z -0.10 m | [0.028, 0.009, 0.907] | [0.025, 0.008, 0.910] | 3.85 mm | 71 | PASS | disp [-0.003, -0.000, -0.098] sign_ok=True ori_drift=0.53deg | [img](stage3_primitive_report/frames/s1_axis_z-0.10.jpg) |
| 1 | position | random target 0 | [0.031, 0.028, 0.906] | [0.028, 0.028, 0.908] | 3.64 mm | 72 | PASS |  | [img](stage3_primitive_report/frames/s1_random_0.jpg) |
| 1 | position | random target 1 | [-0.090, 0.074, 1.083] | [-0.089, 0.072, 1.080] | 3.65 mm | 86 | PASS |  | [img](stage3_primitive_report/frames/s1_random_1.jpg) |
| 1 | position | random target 2 | [-0.043, 0.157, 1.059] | [-0.041, 0.154, 1.057] | 3.46 mm | 91 | PASS |  | [img](stage3_primitive_report/frames/s1_random_2.jpg) |
| 1 | position | random target 3 | [-0.084, 0.022, 0.977] | [-0.083, 0.021, 0.977] | 1.20 mm | 76 | PASS |  | [img](stage3_primitive_report/frames/s1_random_3.jpg) |
| 1 | position | random target 4 | [-0.086, 0.114, 1.112] | [-0.086, 0.112, 1.109] | 3.59 mm | 93 | PASS |  | [img](stage3_primitive_report/frames/s1_random_4.jpg) |
| 1 | position | target 5 cm below table | [0.030, 0.200, 0.750] | [0.039, 0.202, 0.808] | 59.0 mm | 200 | PASS | converged=False contact=True peakF=768.82N finite=True | [img](stage3_primitive_report/frames/s1_below_table.jpg) |
| 1 | orientation | yaw +45 deg | [-179.7, -4.1, 43.3] | [-179.7, -4.2, 42.3] | 1.00 deg | 135 | PASS | pos drift 1.12 mm; x-axis CCW from above=True; readback dyaw=+44.0deg | [img](stage3_primitive_report/frames/s1_yaw+45.jpg) |
| 1 | orientation | yaw -45 deg | [-179.7, -4.1, -46.7] | [-179.6, -4.2, -45.6] | 1.03 deg | 137 | PASS | pos drift 1.12 mm | [img](stage3_primitive_report/frames/s1_yaw-45.jpg) |
| 1 | orientation | yaw +90 deg | [-179.7, -4.1, 88.3] | [-179.7, -4.2, 87.3] | 1.01 deg | 237 | PASS | pos drift 0.59 mm; x-axis CCW from above=True; readback dyaw=+89.0deg | [img](stage3_primitive_report/frames/s1_yaw+90.jpg) |
| 1 | orientation | yaw -90 deg | [-179.7, -4.1, -91.7] | [-179.6, -4.2, -90.6] | 1.05 deg | 236 | PASS | pos drift 0.61 mm | [img](stage3_primitive_report/frames/s1_yaw-90.jpg) |
| 1 | orientation | roll +20 deg | [-159.7, -4.1, -1.7] | [-159.7, -3.9, -1.6] | 0.24 deg | 137 | PASS | pos drift 2.60 mm | [img](stage3_primitive_report/frames/s1_roll+20.jpg) |
| 1 | orientation | roll -20 deg | [-199.7, -4.1, -1.7] | [160.4, -3.9, -1.7] | 0.18 deg | 129 | PASS | pos drift 2.04 mm | [img](stage3_primitive_report/frames/s1_roll-20.jpg) |
| 1 | orientation | pitch +20 deg | [-179.7, 15.9, -1.7] | [-179.7, 15.4, -1.7] | 0.47 deg | 80 | PASS | pos drift 4.31 mm | [img](stage3_primitive_report/frames/s1_pitch+20.jpg) |
| 1 | orientation | pitch -20 deg | [-179.7, -24.1, -1.7] | [-179.6, -24.2, -1.7] | 0.11 deg | 96 | PASS | pos drift 4.71 mm | [img](stage3_primitive_report/frames/s1_pitch-20.jpg) |
| 1 | orientation | combined pos+ori | [0.090, -0.071, 0.935] [-179.7, 5.9, 33.3] | [0.091, -0.071, 0.936] [-179.6, 5.8, 32.3] | 1.17 mm / 1.02 deg | 115 | PASS |  | [img](stage3_primitive_report/frames/s1_combined.jpg) |
| 1 | orientation | round trip (readback->command) | [0.091, -0.071, 0.936] [-179.6, 5.8, 32.3] | [0.090, -0.071, 0.936] [-179.6, 5.8, 32.4] | 0.04 mm / 0.06 deg | 2 | PASS | expect ~zero motion |  |
| 1 | gripper | CLOSE (empty) | width<0.015 | 0.0099 | - | 34 | PASS | object_between=False |  |
| 1 | gripper | MOVE_EEF keeps CLOSED | 0.0099 | 0.0100 | 0.08 mm | 71 | PASS | register=1.0 | [img](stage3_primitive_report/frames/s1_closed_move.jpg) |
| 1 | gripper | OPEN | width>0.07 | 0.0771 | - | 45 | PASS |  |  |
| 1 | lift/desc | DESCEND to z=0.88 | z=0.880 | z=0.882 | xy drift 3.62 mm | 78 | PASS |  |  |
| 1 | lift/desc | LIFT +0.12 m | z=1.002 | z=1.000 | xy drift 1.88 mm | 80 | PASS |  |  |
| 1 | lift/desc | HOLD 50 ticks | drift<2mm | 0.14 mm | 0.14 mm | 50 | PASS |  |  |
| 1 | atomic | ATOMIC dx +1x0.5 x20 | sign +1 on dx | disp [0.028, 0.000, -0.002] m | - | 20 | PASS |  |  |
| 1 | atomic | ATOMIC dx -1x0.5 x20 | sign -1 on dx | disp [-0.028, -0.000, 0.002] m | - | 20 | PASS |  |  |
| 1 | atomic | ATOMIC dy +1x0.5 x20 | sign +1 on dy | disp [-0.001, 0.028, -0.000] m | - | 20 | PASS |  |  |
| 1 | atomic | ATOMIC dy -1x0.5 x20 | sign -1 on dy | disp [-0.000, -0.028, 0.000] m | - | 20 | PASS |  |  |
| 1 | atomic | ATOMIC dz +1x0.5 x20 | sign +1 on dz | disp [-0.003, -0.000, 0.029] m | - | 20 | PASS |  |  |
| 1 | atomic | ATOMIC dz -1x0.5 x20 | sign -1 on dz | disp [0.002, 0.000, -0.031] m | - | 20 | PASS |  |  |
| 1 | atomic | ATOMIC droll +1x0.5 x20 | sign +1 on droll | world rotvec [16.66, -0.02, -3.10] deg | - | 20 | PASS |  |  |
| 1 | atomic | ATOMIC droll -1x0.5 x20 | sign -1 on droll | world rotvec [-16.26, -1.26, 3.54] deg | - | 20 | PASS |  |  |
| 1 | atomic | ATOMIC dpitch +1x0.5 x20 | sign +1 on dpitch | world rotvec [0.06, 12.07, -0.06] deg | - | 20 | PASS |  |  |
| 1 | atomic | ATOMIC dpitch -1x0.5 x20 | sign -1 on dpitch | world rotvec [-0.07, -9.30, 0.09] deg | - | 20 | PASS |  |  |
| 1 | atomic | ATOMIC dyaw +1x0.5 x20 | sign +1 on dyaw | world rotvec [0.17, -0.06, 11.78] deg | - | 20 | PASS |  |  |
| 1 | atomic | ATOMIC dyaw -1x0.5 x20 | sign -1 on dyaw | world rotvec [-0.14, -0.09, -11.81] deg | - | 20 | PASS |  |  |
| 2 | position | axis x +0.05 m | [0.111, -0.024, 1.020] | [0.109, -0.024, 1.021] | 2.12 mm | 58 | PASS | disp [0.048, 0.000, 0.001] sign_ok=True ori_drift=0.23deg | [img](stage3_primitive_report/frames/s2_axis_x+0.05.jpg) |
| 2 | position | axis x -0.05 m | [0.012, -0.024, 1.019] | [0.014, -0.024, 1.019] | 2.17 mm | 58 | PASS | disp [-0.048, -0.000, -0.001] sign_ok=True ori_drift=0.19deg | [img](stage3_primitive_report/frames/s2_axis_x-0.05.jpg) |
| 2 | position | axis x +0.10 m | [0.159, -0.024, 1.021] | [0.158, -0.024, 1.023] | 2.33 mm | 72 | PASS | disp [0.099, 0.000, 0.002] sign_ok=True ori_drift=0.51deg | [img](stage3_primitive_report/frames/s2_axis_x+0.10.jpg) |
| 2 | position | axis x -0.10 m | [-0.038, -0.024, 1.018] | [-0.037, -0.024, 1.017] | 2.03 mm | 73 | PASS | disp [-0.099, 0.000, -0.001] sign_ok=True ori_drift=0.36deg | [img](stage3_primitive_report/frames/s2_axis_x-0.10.jpg) |
| 2 | position | axis y +0.05 m | [0.060, 0.026, 1.022] | [0.059, 0.024, 1.021] | 2.17 mm | 57 | PASS | disp [-0.001, 0.048, -0.001] sign_ok=True ori_drift=0.17deg | [img](stage3_primitive_report/frames/s2_axis_y+0.05.jpg) |
| 2 | position | axis y -0.05 m | [0.060, -0.073, 1.020] | [0.060, -0.071, 1.020] | 2.05 mm | 57 | PASS | disp [-0.001, -0.048, -0.001] sign_ok=True ori_drift=0.20deg | [img](stage3_primitive_report/frames/s2_axis_y-0.05.jpg) |
| 2 | position | axis y +0.10 m | [0.060, 0.075, 1.020] | [0.057, 0.073, 1.018] | 3.90 mm | 71 | PASS | disp [-0.003, 0.098, -0.002] sign_ok=True ori_drift=0.44deg | [img](stage3_primitive_report/frames/s2_axis_y+0.10.jpg) |
| 2 | position | axis y -0.10 m | [0.061, -0.124, 1.020] | [0.058, -0.122, 1.018] | 3.62 mm | 71 | PASS | disp [-0.003, -0.099, -0.002] sign_ok=True ori_drift=0.44deg | [img](stage3_primitive_report/frames/s2_axis_y-0.10.jpg) |
| 2 | position | axis z +0.05 m | [0.060, -0.024, 1.070] | [0.060, -0.024, 1.067] | 2.84 mm | 58 | PASS | disp [0.000, 0.000, 0.047] sign_ok=True ori_drift=0.11deg | [img](stage3_primitive_report/frames/s2_axis_z+0.05.jpg) |
| 2 | position | axis z -0.05 m | [0.060, -0.024, 0.973] | [0.059, -0.024, 0.975] | 2.74 mm | 57 | PASS | disp [-0.001, 0.000, -0.048] sign_ok=True ori_drift=0.22deg | [img](stage3_primitive_report/frames/s2_axis_z-0.05.jpg) |
| 2 | position | axis z +0.10 m | [0.059, -0.024, 1.118] | [0.060, -0.024, 1.116] | 2.77 mm | 74 | PASS | disp [0.001, 0.000, 0.097] sign_ok=True ori_drift=0.19deg | [img](stage3_primitive_report/frames/s2_axis_z+0.10.jpg) |
| 2 | position | axis z -0.10 m | [0.059, -0.024, 0.923] | [0.056, -0.024, 0.925] | 3.65 mm | 71 | PASS | disp [-0.003, 0.000, -0.098] sign_ok=True ori_drift=0.53deg | [img](stage3_primitive_report/frames/s2_axis_z-0.10.jpg) |
| 2 | position | random target 0 | [0.059, 0.070, 0.966] | [0.056, 0.072, 0.964] | 4.45 mm | 135 | PASS |  | [img](stage3_primitive_report/frames/s2_random_0.jpg) |
| 2 | position | random target 1 | [0.106, 0.031, 1.078] | [0.105, 0.030, 1.077] | 1.80 mm | 66 | PASS |  | [img](stage3_primitive_report/frames/s2_random_1.jpg) |
| 2 | position | random target 2 | [-0.091, -0.075, 0.979] | [-0.088, -0.077, 0.981] | 4.84 mm | 274 | PASS |  | [img](stage3_primitive_report/frames/s2_random_2.jpg) |
| 2 | position | random target 3 | [0.022, -0.004, 1.097] | [0.023, -0.005, 1.094] | 3.27 mm | 71 | PASS |  | [img](stage3_primitive_report/frames/s2_random_3.jpg) |
| 2 | position | random target 4 | [-0.086, -0.056, 1.078] | [-0.085, -0.054, 1.075] | 3.42 mm | 88 | PASS |  | [img](stage3_primitive_report/frames/s2_random_4.jpg) |
| 2 | position | target 5 cm below table | [0.061, 0.200, 0.750] | [0.073, 0.209, 0.807] | 58.6 mm | 200 | PASS | converged=False contact=True peakF=543.51N finite=True | [img](stage3_primitive_report/frames/s2_below_table.jpg) |
| 2 | orientation | yaw +45 deg | [178.9, -8.3, 43.7] | [178.8, -8.5, 42.7] | 1.03 deg | 134 | PASS | pos drift 2.41 mm; x-axis CCW from above=True; readback dyaw=+44.0deg | [img](stage3_primitive_report/frames/s2_yaw+45.jpg) |
| 2 | orientation | yaw -45 deg | [178.9, -8.3, -46.3] | [178.9, -8.4, -45.3] | 1.03 deg | 137 | PASS | pos drift 1.52 mm | [img](stage3_primitive_report/frames/s2_yaw-45.jpg) |
| 2 | orientation | yaw +90 deg | [178.9, -8.3, 88.7] | [178.9, -8.4, 87.6] | 1.03 deg | 236 | PASS | pos drift 0.37 mm; x-axis CCW from above=True; readback dyaw=+89.0deg | [img](stage3_primitive_report/frames/s2_yaw+90.jpg) |
| 2 | orientation | yaw -90 deg | [178.9, -8.3, -91.3] | [178.8, -8.3, -90.3] | 1.03 deg | 236 | PASS | pos drift 0.31 mm | [img](stage3_primitive_report/frames/s2_yaw-90.jpg) |
| 2 | orientation | roll +20 deg | [198.9, -8.3, -1.3] | [-161.2, -7.9, -1.3] | 0.35 deg | 139 | PASS | pos drift 3.67 mm | [img](stage3_primitive_report/frames/s2_roll+20.jpg) |
| 2 | orientation | roll -20 deg | [158.9, -8.3, -1.3] | [158.9, -8.0, -1.4] | 0.24 deg | 136 | PASS | pos drift 2.57 mm | [img](stage3_primitive_report/frames/s2_roll-20.jpg) |
| 2 | orientation | pitch +20 deg | [178.9, 11.7, -1.3] | [178.9, 11.3, -1.4] | 0.46 deg | 80 | PASS | pos drift 4.35 mm | [img](stage3_primitive_report/frames/s2_pitch+20.jpg) |
| 2 | orientation | pitch -20 deg | [178.9, -28.3, -1.3] | [178.9, -28.3, -1.3] | 0.08 deg | 95 | PASS | pos drift 4.73 mm | [img](stage3_primitive_report/frames/s2_pitch-20.jpg) |
| 2 | orientation | combined pos+ori | [0.121, -0.104, 0.950] [178.9, 1.7, 33.7] | [0.122, -0.104, 0.951] [179.0, 1.6, 32.6] | 1.22 mm / 1.04 deg | 115 | PASS |  | [img](stage3_primitive_report/frames/s2_combined.jpg) |
| 2 | orientation | round trip (readback->command) | [0.122, -0.104, 0.951] [179.0, 1.6, 32.6] | [0.122, -0.104, 0.951] [179.0, 1.6, 32.7] | 0.05 mm / 0.06 deg | 2 | PASS | expect ~zero motion |  |
| 2 | gripper | CLOSE (empty) | width<0.015 | 0.0099 | - | 34 | PASS | object_between=False |  |
| 2 | gripper | MOVE_EEF keeps CLOSED | 0.0099 | 0.0100 | 0.08 mm | 71 | PASS | register=1.0 | [img](stage3_primitive_report/frames/s2_closed_move.jpg) |
| 2 | gripper | OPEN | width>0.07 | 0.0771 | - | 45 | PASS |  |  |
| 2 | lift/desc | DESCEND to z=0.88 | z=0.880 | z=0.882 | xy drift 3.69 mm | 82 | PASS |  |  |
| 2 | lift/desc | LIFT +0.12 m | z=1.002 | z=1.000 | xy drift 2.05 mm | 80 | PASS |  |  |
| 2 | lift/desc | HOLD 50 ticks | drift<2mm | 0.11 mm | 0.11 mm | 50 | PASS |  |  |
| 2 | atomic | ATOMIC dx +1x0.5 x20 | sign +1 on dx | disp [0.028, 0.000, -0.002] m | - | 20 | PASS |  |  |
| 2 | atomic | ATOMIC dx -1x0.5 x20 | sign -1 on dx | disp [-0.028, -0.000, 0.002] m | - | 20 | PASS |  |  |
| 2 | atomic | ATOMIC dy +1x0.5 x20 | sign +1 on dy | disp [-0.000, 0.027, 0.000] m | - | 20 | PASS |  |  |
| 2 | atomic | ATOMIC dy -1x0.5 x20 | sign -1 on dy | disp [-0.001, -0.027, -0.000] m | - | 20 | PASS |  |  |
| 2 | atomic | ATOMIC dz +1x0.5 x20 | sign +1 on dz | disp [-0.003, -0.000, 0.029] m | - | 20 | PASS |  |  |
| 2 | atomic | ATOMIC dz -1x0.5 x20 | sign -1 on dz | disp [0.002, 0.000, -0.030] m | - | 20 | PASS |  |  |
| 2 | atomic | ATOMIC droll +1x0.5 x20 | sign +1 on droll | world rotvec [16.11, -0.93, -2.70] deg | - | 20 | PASS |  |  |
| 2 | atomic | ATOMIC droll -1x0.5 x20 | sign -1 on droll | world rotvec [-16.15, -0.35, 2.44] deg | - | 20 | PASS |  |  |
| 2 | atomic | ATOMIC dpitch +1x0.5 x20 | sign +1 on dpitch | world rotvec [-0.11, 12.11, 0.12] deg | - | 20 | PASS |  |  |
| 2 | atomic | ATOMIC dpitch -1x0.5 x20 | sign -1 on dpitch | world rotvec [0.14, -9.22, -0.24] deg | - | 20 | PASS |  |  |
| 2 | atomic | ATOMIC dyaw +1x0.5 x20 | sign +1 on dyaw | world rotvec [0.22, -0.24, 11.56] deg | - | 20 | PASS |  |  |
| 2 | atomic | ATOMIC dyaw -1x0.5 x20 | sign -1 on dyaw | world rotvec [-0.25, 0.06, -11.62] deg | - | 20 | PASS |  |  |
