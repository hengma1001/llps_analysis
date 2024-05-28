import os
from typing import Dict

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import distances
from tqdm import tqdm


def cal_mini_dist(atomgrp1: mda.AtomGroup, atomgrp2: mda.AtomGroup) -> float:
    dists = distances.distance_array(
        atomgrp1.positions, atomgrp2.positions, box=atomgrp1.dimensions
    )
    return np.min(dists)


def cal_com_dist(atomgrp: mda.AtomGroup) -> np.array:
    positions = np.array(
        [i.atoms.select_atoms("not name H*").center_of_mass() for i in atomgrp.segments]
    )
    dists = distances.self_distance_array(positions, box=atomgrp.dimensions)
    return dists


def cal_chain_mini_dist(protein: mda.AtomGroup) -> np.array:
    n_segs = protein.n_segments
    dist_mat = []
    for i in range(n_segs):
        for j in range(i + 1, n_segs):
            dist_min = cal_mini_dist(
                protein.segments[i].atoms.select_atoms("name CA"),
                protein.segments[j].atoms.select_atoms("name CA"),
            )
            dist_mat.append(dist_min)

    return dist_mat


def cal_chain_dists(
    pdb: str,
    dcd: str,
    starting_frame: int = 0,
    ending_frames: int = 0,
    every_n_frames: int = 1,
    prog_bar: bool = True,
) -> list:
    label = os.path.basename(dcd)[:-4]
    mda_u = mda.Universe(pdb, dcd)

    protein = mda_u.select_atoms("protein")

    dist_dicts = []
    if ending_frames == 0:
        ending_frames = len(mda.trajectory)
    assert starting_frame < ending_frames
    if prog_bar:
        traj = tqdm(mda_u.trajectory[:ending_frames:every_n_frames])
    else:
        traj = mda_u.trajectory[:ending_frames:every_n_frames]
    for ts in traj:
        dist_dict = {
            "label": label,
            "frame": ts.frame,
            "com_dist": cal_com_dist(protein),
            "mini_dist": np.array(cal_chain_mini_dist(protein)),
        }
        dist_dicts.append(dist_dict)

    return dist_dicts


def triu_to_full(upper_triu, lower_triu=None, num_res=None):

    if num_res is None:
        num_res = int(np.ceil((len(upper_triu) * 2) ** 0.5))
    if lower_triu is None:
        lower_triu = upper_triu
    assert len(upper_triu) == len(lower_triu)
    iu1 = np.triu_indices(num_res, 1)

    cm_full = np.zeros((num_res, num_res))
    cm_full[iu1] = upper_triu
    cm_full.T[iu1] = lower_triu
    np.fill_diagonal(cm_full, 1)
    return cm_full
