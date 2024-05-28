import sys

sys.path.append("../")
import pandas as pd

from llps_analysis.utils import cal_chain_dists, triu_to_full

pdb = "VPAVGx30_20230908_3_3_3_6.0nm_chain.pdb"
traj = "VPAVGx30_20230908_3_3_3_6.0nm_chain_replica_67.dcd"

dist_dict = cal_chain_dists(pdb, traj, ending_frames=100, every_n_frames=2)

df = pd.DataFrame(dist_dict)
df.to_pickle("test.pkl")
