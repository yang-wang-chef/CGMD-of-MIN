# -*- coding: utf-8 -*-
# Mixture: bottlebrush + recognition axle (no ring)
# atom_style full, no def

import numpy as np
import itertools as it
np.random.seed(12345)

# ============== 盒子与通用参数 ==============
rho0             = 0.0005   # 目标初始数密度 (LJ 单位)
bond_len         = 1.0
pad_between      = 2.0      # 分子间安全间隙
max_place_trials = 5000

# ============== Bottlebrush 参数 ==============
n_bottlebrush   = 20
BB_N_backbone   = 40         # 主链珠数
BB_side_len     = 3          # 侧链总珠数 含末端
BB_every_graft  = 5          # 每隔多少个主链珠接一条侧链
BB_graft_offset = 2          # 从第几个主链珠开始接
BB_alternate    = True       # 侧链方向交替

# ============== Atom type 约定 ==============
TYPE_AXLE      = 1           # 通用骨架
TYPE_STOP      = 2           # stop 珠
TYPE_XLINK_RX  = 3           # 识别分子端的交联点
TYPE_XLINK_BB  = 4           # bottlebrush 侧链末端
TYPE_CENTER    = 5           # 识别位点 center

max_type_id = TYPE_CENTER

masses = {i:1.0 for i in range(1, max_type_id+1)}
pair_eps = {i:1.0 for i in range(1, max_type_id+1)}
pair_sig = {i:1.0 for i in range(1, max_type_id+1)}
pair_sig[TYPE_STOP] = 1.20   # stopper 稍微大一点

bond_k  = 300.0
bond_r0 = bond_len

# ============== 识别分子 (原 rotaxane 去掉 ring) 参数 ==============
m_rotaxane = 48#0              # 分子数
RX_n_axle_left  = 8
RX_n_axle_right = 8
RX_left_extra_stopper_n = 3  # 左轴上某一颗是 stopper
RX_graft_len = 6             # 从 center 长出的链长

# ============== 容器与计数器 ==============
atoms  = []  # (aid, mol, type, q, x, y, z)
bonds  = []  # (bid, btype, i, j)
angles = []  # (angid, atype, i, j, k)
aid = 0
bid = 0
angid = 0
mol = 0

placed_centers = []   # 每个分子的几何中心
placed_radii  = []    # 每个分子的近似半径

# ============== 模板一: bottlebrush 局部坐标，居中在原点 ==============
bb_atoms_local = []   # (type, x, y, z)
bb_bonds_local = []   # (i, j)
bb_angles_local = []  # (i, j, k) 用 angle type 1

# 主链沿 x 放置
bb_ids = []
for i in range(BB_N_backbone):
    pos = np.array([i*bond_len, 0.0, 0.0])
    bb_ids.append(len(bb_atoms_local)+1)
    bb_atoms_local.append((TYPE_AXLE, pos[0], pos[1], pos[2]))

for i in range(BB_N_backbone-1):
    bb_bonds_local.append((bb_ids[i], bb_ids[i+1]))
for i in range(1, BB_N_backbone-1):
    bb_angles_local.append((bb_ids[i-1], bb_ids[i], bb_ids[i+1]))

# 侧链接枝
bb_graft_sites = []
for i in range(BB_N_backbone):
    if (i - BB_graft_offset) % BB_every_graft == 0 and 0 < i < BB_N_backbone-1:
        bb_graft_sites.append(i)

for gi, i in enumerate(bb_graft_sites):
    base_id = bb_ids[i]
    base_pos = np.array([
        bb_atoms_local[base_id-1][1],
        bb_atoms_local[base_id-1][2],
        bb_atoms_local[base_id-1][3],
    ])
    sgn = 1.0 if (not BB_alternate or gi % 2 == 0) else -1.0
    gdir = np.array([0.0, 0.0, sgn])
    prev = base_id
    first_new = None
    for n in range(BB_side_len):
        is_last = (n == BB_side_len-1)
        t = TYPE_XLINK_BB if is_last else TYPE_AXLE
        pos = base_pos + gdir*((n+1)*bond_len)
        bb_atoms_local.append((t, pos[0], pos[1], pos[2]))
        new_id = len(bb_atoms_local)
        bb_bonds_local.append((prev, new_id))
        if n == 0:
            bb_angles_local.append((bb_ids[i-1], base_id, new_id))
            bb_angles_local.append((bb_ids[i+1], base_id, new_id))
            first_new = new_id
        prev = new_id
    if BB_side_len >= 2:
        bb_angles_local.append((base_id, first_new, first_new+1))

# 居中
bb_coords = np.array([[a[1],a[2],a[3]] for a in bb_atoms_local], float)
bb_center = bb_coords.mean(axis=0)
bb_coords -= bb_center
bb_atoms_local = [
    (bb_atoms_local[i][0], bb_coords[i,0], bb_coords[i,1], bb_coords[i,2])
    for i in range(len(bb_atoms_local))
]
bb_radius = np.max(np.linalg.norm(bb_coords, axis=1)) + 0.5

# ============== 模板二: “识别轴”局部坐标 (无 ring)，居中在原点 ==============
rx_atoms_local_types = []   # 类型或占位字符串
rx_coords_local = []        # 坐标
rx_bonds_local = []

# center 放在原点
rx_atoms_local_types.append("CENTER")
rx_coords_local.append([0.0, 0.0, 0.0])
center_local_id = len(rx_atoms_local_types)

# 右侧轴与 stopper
prev = center_local_id
for n in range(1, RX_n_axle_right+1):
    rx_atoms_local_types.append(TYPE_AXLE)
    rx_coords_local.append([n*bond_len, 0.0, 0.0])
    new_id = len(rx_atoms_local_types)
    rx_bonds_local.append((prev, new_id))
    prev = new_id

rx_atoms_local_types.append(TYPE_STOP)
rx_coords_local.append([(RX_n_axle_right+1)*bond_len, 0.0, 0.0])
rx_bonds_local.append((prev, len(rx_atoms_local_types)))

# 左侧轴 + 中间一个 stopper + 最端交联点
prev = center_local_id
for n in range(1, RX_n_axle_left+1):
    t = TYPE_STOP if n == RX_left_extra_stopper_n else TYPE_AXLE
    rx_atoms_local_types.append(t)
    rx_coords_local.append([-n*bond_len, 0.0, 0.0])
    new_id = len(rx_atoms_local_types)
    rx_bonds_local.append((prev, new_id))
    prev = new_id

rx_atoms_local_types.append(TYPE_XLINK_RX)
rx_coords_local.append([-(RX_n_axle_left+1)*bond_len, 0.0, 0.0])
rx_bonds_local.append((prev, len(rx_atoms_local_types)))

# 从 center 生长一条 graft 链 (识别链)
base_local = center_local_id
base_pos = np.array(rx_coords_local[base_local-1])
gdir = np.array([0.0, 1.0, 0.0])   # 固定向 +y

prev = base_local
for n in range(RX_graft_len):
    is_last = (n == RX_graft_len-1)
    t = TYPE_XLINK_RX if is_last else TYPE_AXLE
    rx_atoms_local_types.append(t)
    rx_coords_local.append((base_pos + gdir*((n+1)*bond_len)).tolist())
    new_id = len(rx_atoms_local_types)
    rx_bonds_local.append((prev, new_id))
    prev = new_id

# 居中
rx_coords = np.array(rx_coords_local, float)
rx_center = rx_coords.mean(axis=0)
rx_coords -= rx_center
rx_radius = np.max(np.linalg.norm(rx_coords, axis=1)) + 0.5

# ============== 根据总珠子数与目标密度确定 BOX_L ==============
n_per_bb = len(bb_atoms_local)
n_per_rx = len(rx_atoms_local_types)
N_total  = n_bottlebrush * n_per_bb + m_rotaxane * n_per_rx

V = N_total / rho0
BOX_L = V ** (1.0/3.0)
BOX_L *= 1.02   # 稍微放大一点

print(f"总珠子数: {N_total}, 目标密度: {rho0}, 盒长: {BOX_L:.3f}")

# ============== 放置 bottlebrush ==============
for m in range(n_bottlebrush):
    mol += 1

    placed = False
    trials = 0
    while not placed and trials < max_place_trials:
        trials += 1
        # 随机旋转矩阵
        q = np.random.normal(size=4); q = q/np.linalg.norm(q)
        w,xq,yq,zq = q
        Rm = np.array([
            [1-2*(yq*yq+zq*zq), 2*(xq*yq - zq*w),   2*(xq*zq + yq*w)],
            [2*(xq*yq + zq*w),  1-2*(xq*xq+zq*zq), 2*(yq*zq - xq*w)],
            [2*(xq*zq - yq*w),  2*(yq*zq + xq*w),  1-2*(xq*xq+yq*yq)]
        ])
        center = np.random.uniform(low=pad_between, high=BOX_L-pad_between, size=3)

        # 边界判定
        if np.any(center - bb_radius < 0.0) or np.any(center + bb_radius > BOX_L):
            continue

        # 与已放分子保持距离
        ok = True
        for c, r in zip(placed_centers, placed_radii):
            if np.linalg.norm(center - c) < (bb_radius + r + pad_between):
                ok = False
                break
        if not ok:
            continue

        placed = True

    if not placed:
        raise RuntimeError("放置 bottlebrush 失败，请增大 BOX_L 或减小数量")

    local_to_global = []
    for t,x,y,z in bb_atoms_local:
        gx,gy,gz = (Rm @ np.array([x,y,z])) + center
        aid += 1
        atoms.append((aid, mol, t, 0.0, gx, gy, gz))
        local_to_global.append(aid)

    for i,j in bb_bonds_local:
        bid += 1
        bonds.append((bid, 1, local_to_global[i-1], local_to_global[j-1]))

    for i,j,k in bb_angles_local:
        angid += 1
        angles.append((angid, 1, local_to_global[i-1], local_to_global[j-1], local_to_global[k-1]))

    placed_centers.append(center)
    placed_radii.append(bb_radius)

# ============== 放置识别轴分子 (无 ring) ==============
for rxi in range(m_rotaxane):
    mol += 1

    # 随机旋转和平移
    placed = False
    trials = 0
    while not placed and trials < max_place_trials:
        trials += 1
        q = np.random.normal(size=4); q = q/np.linalg.norm(q)
        w,xq,yq,zq = q
        Rm = np.array([
            [1-2*(yq*yq+zq*zq), 2*(xq*yq - zq*w),   2*(xq*zq + yq*w)],
            [2*(xq*yq + zq*w),  1-2*(xq*xq+zq*zq), 2*(yq*zq - xq*w)],
            [2*(xq*zq - yq*w),  2*(yq*zq + xq*w),  1-2*(xq*xq+yq*yq)]
        ])
        center = np.random.uniform(low=pad_between, high=BOX_L-pad_between, size=3)

        if np.any(center - rx_radius < 0.0) or np.any(center + rx_radius > BOX_L):
            continue

        ok = True
        for c, r in zip(placed_centers, placed_radii):
            if np.linalg.norm(center - c) < (rx_radius + r + pad_between):
                ok = False
                break
        if not ok:
            continue

        placed = True

    if not placed:
        raise RuntimeError("放置识别分子失败，请增大 BOX_L 或减小数量")

    local_to_global = []
    for idx,tloc in enumerate(rx_atoms_local_types, start=1):
        if tloc == "CENTER":
            t = TYPE_CENTER
        else:
            t = tloc
        x,y,z = rx_coords[idx-1]
        gx,gy,gz = (Rm @ np.array([x,y,z])) + center
        aid += 1
        atoms.append((aid, mol, t, 0.0, gx, gy, gz))
        local_to_global.append(aid)

    # 键
    for i,j in rx_bonds_local:
        bid += 1
        bonds.append((bid, 1, local_to_global[i-1], local_to_global[j-1]))

    # 利用邻接关系生成所有角 (type 1)
    first_id = local_to_global[0]
    last_id  = local_to_global[-1]
    adj = {i:[] for i in range(first_id, last_id+1)}
    for _,_,ii,jj in bonds:
        if first_id <= ii <= last_id and first_id <= jj <= last_id:
            adj[ii].append(jj)
            adj[jj].append(ii)

    seen = set()
    for j in range(first_id, last_id+1):
        nbrs = sorted(adj.get(j, []))
        for i,k in it.combinations(nbrs, 2):
            key1 = (i,j,k)
            key2 = (k,j,i)
            if key1 in seen or key2 in seen:
                continue
            angid += 1
            angles.append((angid, 1, i, j, k))
            seen.add(key1)
            seen.add(key2)

    placed_centers.append(center)
    placed_radii.append(rx_radius)

# ============== 写 mixture.data ==============
xlo,ylo,zlo = 0.0, 0.0, 0.0
xhi,yhi,zhi = BOX_L, BOX_L, BOX_L

n_atoms = len(atoms)
n_bonds = len(bonds)
n_angles = len(angles)
n_atom_types = max_type_id

with open("mixture-no-ring-8-test.data", "w") as f:
    f.write("Mixture of bottlebrush and recognition axles (no ring), atom_style full\n\n")
    f.write(f"{n_atoms} atoms\n")
    f.write(f"{n_bonds} bonds\n")
    f.write(f"{n_angles} angles\n\n")

    f.write(f"{n_atom_types} atom types\n")
    f.write("2 bond types\n")
    f.write("1 angle types\n\n")
    f.write(f"{xlo:.3f} {xhi:.3f} xlo xhi\n")
    f.write(f"{ylo:.3f} {yhi:.3f} ylo yhi\n")
    f.write(f"{zlo:.3f} {zhi:.3f} zlo zhi\n\n")

    f.write("Masses\n\n")
    for t in range(1, n_atom_types+1):
        f.write(f"{t} {masses.get(t,1.0):.6f}\n")
    f.write("\n")

    f.write("Atoms  # atom_style full\n\n")
    for a in atoms:
        f.write(f"{a[0]} {a[1]} {a[2]} 0.0000 {a[4]:.6f} {a[5]:.6f} {a[6]:.6f}\n")
    f.write("\n")

    f.write("Bonds\n\n")
    for b in bonds:
        f.write(f"{b[0]} {b[1]} {b[2]} {b[3]}\n")
    f.write("\n")

    f.write("Angles\n\n")
    for g in angles:
        f.write(f"{g[0]} {g[1]} {g[2]} {g[3]} {g[4]}\n")

print("写出完成: mixture.data")

#%% ============================================
#%% ========= 生成 in.lj 内容 =========
temperature   = 1.0
gamma         = 2.0
timestep      = 0.01
dump_stride   = 10000

OUT_IN    = "input-no-ring-8-test.in"

lines = []
lines.append("# Auto generated LAMMPS input (no ring)\n")
lines.append("units           lj\n")
lines.append("atom_style      full\n")
lines.append("bond_style      hybrid morse fene\n")
lines.append("angle_style     cosine\n")
lines.append("boundary        p p p\n")
lines.append("read_data      mixture-no-ring-8-test.data\n")


lines.append("\n# Neighbor\n")
lines.append("neighbor        0.3 bin\n")
lines.append("neigh_modify    every 1 delay 0 check yes\n")

lines.append("\n# Nonbonded\n")
lines.append("pair_style      lj/cut {}\n".format(2**(1/6)))
lines.append("pair_modify     mix arithmetic shift yes\n")
lines.append("special_bonds   lj 0 1 1\n")

lines.append("\n# Default pair coeffs\n")
lines.append("pair_coeff      * * 1.0 1.0 {}\n".format(2**(1/6)))
lines.append("pair_coeff      2 * 1.0 2.0 {}\n".format(2*2**(1/6)))  # 尽量一致，stopper也保留！！

lines.append("\n# Bonds and angles\n")
lines.append("bond_coeff      1 fene 30.0 1.5 1.0 1.0\n")
lines.append("bond_coeff      2 morse 50 3.58 0.96\n")
lines.append("angle_coeff     1 1.0\n")

lines.append("\n# Groups\n")
lines.append("group           axle      type 1\n")
lines.append("group           stopper   type 2\n")
lines.append("group           rx_xlink  type 3\n")
lines.append("group           bb_end    type 4\n")
lines.append("group           centers   type 5\n")


lines.append("\n# Output\n")

lines.append("variable        TH equal 1.0\n")
lines.append('velocity         all create ${TH} 247238746\n')
lines.append('minimize         1e-8 1e-8 100000 100000\n')


lines.append("\n# Integrator and thermostat\n")
lines.append("thermo_style     custom step temp lx ly lz density press pxx pyy pzz pe ke etotal ebond eangle epair\n")
lines.append("thermo           1000\n")
lines.append(f"timestep         {timestep}\n")
lines.append("fix             1 all npt temp ${TH} ${TH} 1 iso 0.1 0.1 5 drag 2\n")
lines.append("dump            dp all custom 5000 traj.lammpstrj id type mol xu yu zu\n")
lines.append("run             100000\n")
lines.append("unfix           1\n")
lines.append("undump          dp\n")
lines.append("write_data      npt.data\n")

lines.append("\n# Shrink the model to Crosslink with fix/react placeholder\n")

lines.append("reset_timestep  0\n")

lines.append('variable        den equal "density"\n')
lines.append('variable        lx0 equal "lx"\n')
lines.append('variable        ly0 equal "ly"\n')
lines.append('variable        lz0 equal "lz"\n')
lines.append("variable        lf equal (((${den}*${lx0}*${ly0}*${lz0})/0.6)^(1/3))\n")
lines.append("fix             00 all nve\n")
lines.append("fix             11 all temp/berendsen ${TH} ${TH} 0.5\n")
lines.append("fix             22 all deform 1 x final 0 ${lf} y final 0 ${lf} z final 0 ${lf}\n")
lines.append("dump            11 all custom 1000 shrink.lammpstrj id mol type xu yu zu\n")
lines.append("run             100000\n")
lines.append("unfix           00\n")
lines.append("unfix           11\n")
lines.append("unfix           22\n")
lines.append("undump          11\n")
lines.append("write_data      deformed06.data nocoeff \n")

###################################################################
lines.append("\n# Crosslinking with fix/react placeholder\n")

lines.append('variable        linker atom "type==4"\n')
lines.append("group           linker dynamic all var linker\n")
lines.append("variable        nf equal count(linker)\n")
lines.append('variable        N_initial equal "bonds"\n')
lines.append("variable        N equal ${N_initial}\n")
lines.append('variable        Ncl equal "bonds-v_N"\n')
lines.append("variable        s equal step\n")

lines.append("variable        F equal 0.01\n")
lines.append("variable        B equal 0\n")
lines.append("molecule        mol1 pre-molecule.data\n")
lines.append("molecule        mol2 post-molecule.data\n")


lines.append("reset_timestep  0\n")
lines.append("fix rxns all bond/react stabilization yes statted_grp .03 reset_mol_ids yes&\n")
lines.append("  react rxn1_stp1 all 100 0.0 1.1 mol1 mol2 automap-F.data prob ${F} 12345 max_rxn 1044 &\n")
lines.append("  react rxn1_stp1 all 100 0.0 1.1 mol2 mol1 automap-B.data prob ${B} 12345\n")
lines.append("comm_modify cutoff 5.0\n")
lines.append("variable        DoC equal v_Ncl/v_nf\n")
lines.append('variable        doc equal "v_DoC"\n')
lines.append("variable        c1 equal f_rxns[1]\n")
lines.append("variable        c2 equal f_rxns[2]\n")
lines.append("thermo_style    custom step temp etotal ke pe ebond eangle evdwl press vol lx ly lz density f_rxns[1] f_rxns[2] v_DoC\n")
lines.append("fix             int all nve/limit 0.05\n")
lines.append("fix             lang all langevin ${{TH}} ${{TH}} {} 936931 tally no\n".format(gamma))
lines.append("dump            11 all custom 1000 crosslink-wrap.lammpstrj id mol type xu yu zu\n")
lines.append("run             1000000\n")
lines.append("write_data      Final.data nocoeff\n") 
lines.append("unfix           int\n")
lines.append("unfix           lang\n")
lines.append("undump          11\n")


open(OUT_IN, "w", encoding="utf-8").write("".join(lines))
print(f"写出完成: {OUT_IN}")


