import os
from pathlib import Path

# current_working_dir = Path(os.getcwd())
# print(current_working_dir)

work_dir = Path(os.environ["HOME"].replace("home", "work"))
checkpoint_dir = "saved_models"
try:
    if os.environ["WHEREAMI"] == "cluster":
        checkpoint_dir = work_dir / checkpoint_dir
except:
    pass

current_dir = os.getcwd().split("/")[-1]
app = "../" if current_dir == "notebooks" else ""

DISTANCE_MATRIX_PATH = app + "src/data/human_LeftParcelGeodesicDistmat.txt"
DISTANCE_MATRIX_PATH_MOUSE = app + "src/data/mouse_ccf_isocortex_distance_matrix_sampled_43_no_first_row_col.npy"

CORTICAL_AREAS_PATH = app + "src/data/human_areaNamesGlasser180.txt"
CORTICAL_AREAS_PATH_MOUSE = app + "src/data/mouse_area_names.txt"

COG_NETWORK_OVERLAP = app + "src/data/hcp_regions_cog_networks_overlap.csv"

SPINE_COUNT_MOUSE = app + "src/data/mouse_spine_counts.mat"
SPINE_COUNT_HUMAN = app + "src/data/human_myelin_HCP_vec.mat"

DMN_AREAS_PATH = app + "src/data/dmn_areas.txt"
CORE_DMN_AREAS_PATH = app + "src/data/core_dmn_areas.txt"