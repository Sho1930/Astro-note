import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import solve_ivp

# ========================
# ユーザー定義パラメータ
# ========================
config = 'p'       # 'p': prolate object, 'o': oblate object
record = False     # True: 動画を保存, False: 保存しない
divider = 50       # アニメーション更新間隔

# ------------------------
# 数値積分設定
tfinal = 10       # 終了時刻 [s]
dt = 1e-3         # タイムステップ [s]

if config == 'p':
    w  = 0.5
    h  = 2
    wo = 1
    w3 = 5
elif config == 'o':
    w  = 4
    h  = 0.5
    wo = 4
    w3 = 5
else:
    raise ValueError("Invalid config: choose 'p' or 'o'")

m  = 1
IT = (1/12) * m * (h**2 + w**2)
I3 = (1/6) * m * w**2
wn = (1 - I3/IT) * w3
theta = np.arccos( I3 * w3 / np.sqrt((IT * wo)**2 + (I3 * w3)**2) )
para = {'IT': IT, 'I3': I3, 'wo': wo, 'w3': w3, 'theta': theta}

# ------------------------
# 微分方程式 (体対称自由運動)
#   ψ' = wo/sinθ,   φ' = wn
def torque_free_motion_symmetric_top(t, X, para):
    wo = para['wo']
    w3 = para['w3']
    IT = para['IT']
    I3 = para['I3']
    theta = para['theta']
    wn = (1 - I3/IT) * w3
    dpsi = wo / np.sin(theta)
    dphi = wn
    return [dpsi, dphi]

X0 = [0, 0]
tspan = (0, tfinal)
t_eval = np.arange(0, tfinal+dt, dt)
sol = solve_ivp(lambda t, X: torque_free_motion_symmetric_top(t, X, para),
                tspan, X0, t_eval=t_eval, rtol=1e-6, atol=1e-6)
psi = sol.y[0]
phi = sol.y[1]
tt = t_eval

# ========================
# 初期描画用設定
qlw = 2
len_val = 2
e1 = np.array([len_val, 0, 0])
e2 = np.array([0, len_val, 0])
hG = np.array([0, 0, 3*len_val])

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
# 慣性系軸
ax.quiver(0,0,0, e1[0], e1[1], e1[2], color='r', linestyle='--', linewidth=0.5*qlw)
ax.quiver(0,0,0, e2[0], e2[1], e2[2], color='r', linestyle='--', linewidth=0.5*qlw)
ax.quiver(0,0,0, hG[0], hG[1], hG[2], color='r', linewidth=qlw)
ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_zlabel('$z$')
ax.set_xlim([-6,6]); ax.set_ylim([-6,6]); ax.set_zlim([-3,6])
ax.view_init(elev=20, azim=30)
ax.grid(True)

# -------------------------
# 3-2-3 回転行列の定義
def C_C_B(phi):
    return np.array([[np.cos(phi), -np.sin(phi), 0],
                     [np.sin(phi),  np.cos(phi), 0],
                     [0,            0,           1]])
def A_C_C(th):
    return np.array([[np.cos(th), 0, np.sin(th)],
                     [0,          1, 0],
                     [-np.sin(th),0, np.cos(th)]])
def I_C_A(psi):
    return np.array([[np.cos(psi), -np.sin(psi), 0],
                     [np.sin(psi),  np.cos(psi), 0],
                     [0,            0,           1]])

# -------------------------
# 初期慣性/ボディフレームの初期化
cCB = C_C_B(0)
aCc = A_C_C(theta)
ICa = I_C_A(0)
I_C_B = ICa @ aCc @ cCB
DCM = I_C_B
b1 = DCM @ e1
b2 = DCM @ e2
b3 = DCM @ hG
IwB = DCM @ np.array([-wo, 0, w3])

# ボディ軸
q_b1 = ax.quiver(0,0,0, b1[0], b1[1], b1[2], color='b', linestyle='--', linewidth=0.5*qlw)
q_b2 = ax.quiver(0,0,0, b2[0], b2[1], b2[2], color='b', linestyle='--', linewidth=0.5*qlw)
q_b3 = ax.quiver(0,0,0, b3[0], b3[1], b3[2], color='b', linewidth=qlw)
q_IwB = ax.quiver(0,0,0, IwB[0], IwB[1], IwB[2], color='g', linewidth=2*qlw)

# -------------------------
# 3D ボックス（物体）の作成
x_box = w/2
y_box = x_box
z_box = h/2
vv = np.array([[ x_box, -y_box, -z_box],
               [ x_box,  y_box, -z_box],
               [-x_box,  y_box, -z_box],
               [-x_box, -y_box, -z_box],
               [ x_box, -y_box,  z_box],
               [ x_box,  y_box,  z_box],
               [-x_box,  y_box,  z_box],
               [-x_box, -y_box,  z_box]])
fac = np.array([[0,1,5,4],
                [1,2,6,5],
                [2,3,7,6],
                [3,0,4,7],
                [0,1,2,3],
                [4,5,6,7]])
VV = (DCM @ vv.T).T
box_poly = Poly3DCollection(VV[fac], facecolors=[0.8,0.8,0.8], alpha=0.2)
ax.add_collection3d(box_poly)

# -------------------------
# 三角錐（cone）の生成関数
def generate_cone_poly(u, n, f_scale):
    # u: 基準ベクトル（単位化済み）
    # n: cone の軸となる単位ベクトル
    # mu: 0〜2π で円周上の点を生成
    mu = np.linspace(0, 2*np.pi, 200)
    # MATLABと同様の式：v = (1-cos(mu))*dot(n,u)*n + cos(mu)*u - sin(mu)*cross(n,u)
    v = (1 - np.cos(mu))[:, None]*(np.dot(n, u)*n) + np.cos(mu)[:, None]*u - np.sin(mu)[:, None]*np.cross(n, u)
    # 頂点を組む: apex (0,0,0), 次に u, その後 v の各点
    vertices = np.vstack((np.zeros((1,3)), u.reshape(1,3), v))
    return f_scale * vertices  # スケール調整

# 初期 space cone（H を中心軸, ω を母線）の作成（固定）
# H は慣性系軸 hG（赤矢印）とする
n_space = hG / np.linalg.norm(hG)
u_space = IwB / np.linalg.norm(IwB)
space_cone_verts = generate_cone_poly(u_space, n_space, f_scale=5)
space_cone_poly = Poly3DCollection([space_cone_verts], facecolors='magenta', alpha=0.3)
ax.add_collection3d(space_cone_poly)

# 初期 body cone（b1 を中心軸, ω を母線）の作成
n_body = b3 / np.linalg.norm(b3)
u_body = IwB / np.linalg.norm(IwB)  # MATLABでは u は IwB の単位ベクトル
body_cone_verts = generate_cone_poly(u_body, n_body, f_scale=5)
body_cone_poly = Poly3DCollection([body_cone_verts], facecolors='blue', alpha=0.2)
ax.add_collection3d(body_cone_poly)

ax.tick_params(labelsize=12)

# ========================
# アニメーション更新関数
# ========================
def update(frame):
    i = frame
    current_psi = psi[i]
    current_phi = phi[i]
    # 3-2-3 角の回転行列更新
    cCB = C_C_B(current_phi)
    aCc = A_C_C(theta)
    ICa = I_C_A(current_psi)
    I_C_B = ICa @ aCc @ cCB
    DCM = I_C_B
    # ボディ軸更新
    b1_new = DCM @ e1
    b2_new = DCM @ e2
    b3_new = DCM @ hG
    # IwB 更新: w1 = -wo*cos(wn*t), w2 = wo*sin(wn*t)
    t_current = tt[i]
    w1_val = -wo * np.cos(wn * t_current)
    w2_val = wo * np.sin(wn * t_current)
    IwB_new = DCM @ np.array([w1_val, w2_val, w3])
    
    # 軸クリアと設定
    ax.cla()
    ax.set_xlim([-6,6])
    ax.set_ylim([-6,6])
    ax.set_zlim([-3,6])
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_zlabel('$z$')
    ax.view_init(elev=20, azim=30)
    ax.grid(True)
    
    # 慣性系軸
    ax.quiver(0,0,0, e1[0], e1[1], e1[2], color='r', linestyle='--', linewidth=0.5*qlw)
    ax.quiver(0,0,0, e2[0], e2[1], e2[2], color='r', linestyle='--', linewidth=0.5*qlw)
    ax.quiver(0,0,0, hG[0], hG[1], hG[2], color='r', linewidth=qlw)
    # ボディ軸
    ax.quiver(0,0,0, b1_new[0], b1_new[1], b1_new[2], color='b', linestyle='--', linewidth=0.5*qlw)
    ax.quiver(0,0,0, b2_new[0], b2_new[1], b2_new[2], color='b', linestyle='--', linewidth=0.5*qlw)
    ax.quiver(0,0,0, b3_new[0], b3_new[1], b3_new[2], color='b', linewidth=qlw)
    # IwB
    ax.quiver(0,0,0, IwB_new[0], IwB_new[1], IwB_new[2], color='g', linewidth=2*qlw)
    
    # 回転するボックス更新
    VV_new = (DCM @ vv.T).T
    box_poly_new = Poly3DCollection(VV_new[fac], facecolors='black', alpha=0.5)
    ax.add_collection3d(box_poly_new)
    
    # body cone 更新（b3_new を中心軸, ω を母線）
    n_body_new = b3_new / np.linalg.norm(b3_new)
    u_new = IwB_new / np.linalg.norm(IwB_new)
    body_cone_verts_new = generate_cone_poly(u_new, n_body_new, f_scale=5)
    body_cone_poly_new = Poly3DCollection([body_cone_verts_new], facecolors='cyan', alpha=0.3)
    ax.add_collection3d(body_cone_poly_new)
    
    # space cone は固定（初期のものとして再描画）
    space_cone_poly_new = Poly3DCollection([space_cone_verts], facecolors='magenta', alpha=0.3)
    ax.add_collection3d(space_cone_poly_new)
    
    ax.set_title(f"Time = {t_current:.2f} s")
    return []

# ========================
# アニメーション作成・保存
# ========================
frames = range(0, len(tt), divider)
ani = FuncAnimation(fig, update, frames=frames, blit=False)

if record:
    writer = FFMpegWriter(fps=int(1/dt/divider))
    ani.save("torque_free_cones.mp4", writer=writer)
    print("Saved animation to torque_free_cones.mp4")
else:
    plt.show()








# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, FFMpegWriter
# from mpl_toolkits.mplot3d import Axes3D

# # --- クォータニオンの基本操作 ---
# def quat_mult(q, r):
#     # q, r: [w, x, y, z]
#     w1, x1, y1, z1 = q
#     w2, x2, y2, z2 = r
#     return np.array([
#         w1*w2 - x1*x2 - y1*y2 - z1*z2,
#         w1*x2 + x1*w2 + y1*z2 - z1*y2,
#         w1*y2 - x1*z2 + y1*w2 + z1*x2,
#         w1*z2 + x1*y2 - y1*x2 + z1*w2
#     ])

# def quat_normalize(q):
#     return q / np.linalg.norm(q)

# def quat_to_rot_matrix(q):
#     # q = [w, x, y, z]
#     w, x, y, z = q
#     R = np.array([
#         [1 - 2*(y**2+z**2), 2*(x*y - z*w),   2*(x*z + y*w)],
#         [2*(x*y + z*w),     1 - 2*(x**2+z**2), 2*(y*z - x*w)],
#         [2*(x*z - y*w),     2*(y*z + x*w),   1 - 2*(x**2+y**2)]
#     ])
#     return R

# # --- シミュレーションパラメータ ---
# dt = 0.02       # タイムステップ [s]
# total_time = 20  # 総シミュレーション時間 [s]
# num_steps = int(total_time/dt)

# # 軸対象の剛体（円筒）のトルクフリーモーションの角速度（解析解）
# # 初期条件: ω₁(0)=1, ω₂(0)=0, ω₃=1 とするので、解析解では
# # ω₁(t)=cos(t), ω₂(t)=sin(t), ω₃(t)=1
# def angular_velocity(t):
#     return np.array([np.cos(t), np.sin(t), 1.0])

# # 初期姿勢は単位クォータニオン（回転なし）
# q = np.array([1.0, 0.0, 0.0, 0.0])

# # --- 円筒の形状 ---
# radius = 1.0   # 円筒の半径
# height = 2.0   # 円筒の高さ
# num_points = 30

# # 上面と下面の円周上の点（局所座標系）
# angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
# top_circle = np.array([[radius * np.cos(a), radius * np.sin(a), height/2] for a in angles])
# bottom_circle = np.array([[radius * np.cos(a), radius * np.sin(a), -height/2] for a in angles])

# # --- matplotlib 3D プロット設定 ---
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlim([-3, 3])
# ax.set_ylim([-3, 3])
# ax.set_zlim([-3, 3])
# ax.set_box_aspect([1,1,1])
# ax.set_title("Torque-Free Motion of a Cylinder")

# # グローバルなシミュレーション時刻
# time_elapsed = 0.0

# def init():
#     ax.cla()
#     ax.set_xlim([-3, 3])
#     ax.set_ylim([-3, 3])
#     ax.set_zlim([-3, 3])
#     ax.set_box_aspect([1,1,1])
#     ax.set_title("Torque-Free Motion of a Cylinder")
#     return []

# def update(frame):
#     global q, time_elapsed
#     time_elapsed += dt
    
#     # 現在時刻の角速度（解析解）
#     omega = angular_velocity(time_elapsed)
#     # 角速度を純虚数クォータニオンに変換: (0, ω₁, ω₂, ω₃)
#     omega_quat = np.concatenate(([0.0], omega))
    
#     # クォータニオン微分: dq/dt = 0.5 * q ⊗ omega_quat
#     dq = 0.5 * quat_mult(q, omega_quat)
#     q = q + dq * dt
#     q = quat_normalize(q)
    
#     # 現在の姿勢から回転行列を取得
#     R = quat_to_rot_matrix(q)
    
#     # 局所座標系の円筒の各点に回転を適用
#     top_rot = (R @ top_circle.T).T   # (num_points, 3)
#     bottom_rot = (R @ bottom_circle.T).T
    
#     # プロットをクリアして再描画
#     ax.cla()
#     ax.set_xlim([-3, 3])
#     ax.set_ylim([-3, 3])
#     ax.set_zlim([-3, 3])
#     ax.set_box_aspect([1,1,1])
#     ax.set_title("Torque-Free Motion of a Cylinder")
    
#     # 上面・下面の円周を描画
#     ax.plot(top_rot[:,0], top_rot[:,1], top_rot[:,2], color='b')
#     ax.plot(bottom_rot[:,0], bottom_rot[:,1], bottom_rot[:,2], color='b')
#     # 上面と下面を結ぶ縦線を描画
#     for i in range(num_points):
#         xs = [top_rot[i,0], bottom_rot[i,0]]
#         ys = [top_rot[i,1], bottom_rot[i,1]]
#         zs = [top_rot[i,2], bottom_rot[i,2]]
#         ax.plot(xs, ys, zs, color='b')
    
#     return []

# # アニメーションの作成
# ani = FuncAnimation(fig, update, frames=num_steps, init_func=init,
#                     interval=dt*1000, blit=False)

# # MP4形式で保存する設定（FFMpegWriter を使用）
# writer = FFMpegWriter(fps=1/dt)

# # アニメーションを MP4 ファイルとして保存
# ani.save("torque_free_cylinder.mp4", writer=writer)
# print("Saved animation to torque_free_cylinder.mp4")

# plt.show()
