import os
from zipfile import ZipFile
from io import BytesIO
from urllib.request import urlopen
import argparse

# Local imports
from cfat.main import CFAT_algorithm

def check_study_dataset_IK_dir(DIRNAME_STUDY_IK):
    if not os.path.exists(DIRNAME_STUDY_IK):
        download_datasets = input(
            "Could not find reference to the raw IK data included in the ISO-VR-Pointing Dataset. Do you want to download it (~130MB after unpacking)? (y/N) ")
        if download_datasets.lower().startswith("y"):
            print(f"Will download and unzip to '{DIRNAME_STUDY_IK}'.")
            print("Downloading archive... ", end='', flush=True)
            resp = open("/home/florian/mpc-mujoco-git/mpc_mujoco/data/ISO_VR_Pointing_IK_Raw.zip", 'rb')
            #resp = urlopen("http://zenodo.org/ISO_VR_Pointing_IK_Raw.zip")
            zipfile = ZipFile(BytesIO(resp.read()))
            print("unzip archive... ", end='', flush=True)
            for file in zipfile.namelist():
                zipfile.extract(file, path=DIRNAME_STUDY_IK)
            print("done.")
            assert os.path.exists(DIRNAME_STUDY_IK), "Internal Error during unpacking of ISO-VR-Pointing Dataset."
        else:
            raise FileNotFoundError("Please ensure that 'DIRNAME_STUDY' points to a valid directory containing the ISO-VR-Pointing Dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CFAT on an Inverse Kinematics file from the ISO-VR-Pointing Dataset.')
    parser.add_argument('--dirname', dest='DIRNAME_STUDY_IK', default='study_IK_raw/', help='Directory path of the raw IK data included in the ISO-VR-Pointing Dataset.')
    parser.add_argument('--username', dest='username', default='U1', help='Username (U1-U6); used for model file (and table_filename, if not specified).')
    parser.add_argument('--task_condition', dest='task_condition', default='T_Pose', help='Task condition; used for table_filename.')
    parser.add_argument('--table_filename', dest='table_filename', help='Filename to run CFAT with.')
    parser.add_argument('--use_mujoco', dest='use_mujoco_py', action='store_false', help='Whether to use MuJoCo Python bindings instead of mujoco-py.')
    args = parser.parse_args()

    DIRNAME_STUDY_IK = args.DIRNAME_STUDY_IK
    check_study_dataset_IK_dir(DIRNAME_STUDY_IK)

    username = args.username
    task_condition = args.task_condition  #"Standing_ID_ISO_15_plane"

    if args.table_filename is not None:
        table_filename = args.table_filename
    else:
        table_filename = os.path.join(DIRNAME_STUDY_IK, f"{username}_{task_condition}.mot")

    model_filename = f"models/OriginExperiment_{username}.xml"

    physical_joints = ["elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion", "pro_sup", "deviation", "flexion"]
    param_t_activation = 0.04
    param_t_excitation = 0.03

    timestep = 0.002  # in seconds
    results_dir = f"_results/{username}_{timestep}s/"

    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    CFAT_algorithm(table_filename,
                   model_filename,
                   physical_joints=physical_joints,
                   param_t_activation=param_t_activation,
                   param_t_excitation=param_t_excitation,
                   num_target_switches=None,
                   ensure_constraints=False,
                   reset_pos_and_vel=False,
                   useexcitationcontrol=True,
                   optimize_excitations=False,
                   use_qacc=True,
                   timestep=timestep,
                   results_dir=results_dir,
                   use_mujoco_py=args.use_mujoco_py)