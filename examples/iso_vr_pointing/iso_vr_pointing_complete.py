import os
from pathlib import Path
import logging
from zipfile import ZipFile
from io import BytesIO
from urllib.request import urlopen
from cfat.main import CFAT_algorithm

def check_study_dataset_IK_dir(DIRNAME_STUDY_IK):
    if not os.path.exists(DIRNAME_STUDY_IK):
        download_datasets = input(
            "Could not find reference to the raw IK data included in the ISO-VR-Pointing Dataset. Do you want to download it (~130MB after unpacking)? (y/N) ")
        if download_datasets.lower().startswith("y"):
            print(f"Will download and unzip to '{os.path.abspath(DIRNAME_STUDY_IK)}'.")
            print("Downloading archive... ", end='', flush=True)
            resp = urlopen("https://zenodo.org/record/7300062/files/ISO_VR_Pointing_IK_Raw.zip?download=1")
            zipfile = ZipFile(BytesIO(resp.read()))
            print("unzip archive... ", end='', flush=True)
            for file in zipfile.namelist():
                zipfile.extract(file, path=DIRNAME_STUDY_IK)
            print("done.")
            assert os.path.exists(DIRNAME_STUDY_IK), "Internal Error during unpacking of ISO-VR-Pointing Dataset."
        else:
            raise FileNotFoundError("Please ensure that 'DIRNAME_STUDY_IK' points to a valid directory containing the raw IK data included in the ISO-VR-Pointing Dataset.")


if __name__ == "__main__":
    # Change current working directory to file directory
    os.chdir(Path(__file__).parent)
    logging.info(Path(__file__).parent)

    DIRNAME_STUDY_IK = "study_IK_raw/"
    check_study_dataset_IK_dir(DIRNAME_STUDY_IK)

    print('\n\n                 +++ Computation of Feasible Applied Torques (CFAT) +++')
    filelist = [(username, os.path.abspath(os.path.join(DIRNAME_STUDY_IK, f))) for username in [f"U{i}" for i in range(1, 7)]
                for f in os.listdir(DIRNAME_STUDY_IK) if
                os.path.isfile(os.path.abspath(os.path.join(DIRNAME_STUDY_IK, f))) and
                f.startswith(username) and f.endswith('.mot')]

    CFAT_timestep = 0.002

    physical_joints = ["elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion", "pro_sup", "deviation", "flexion"]
    param_t_activation = 0.04
    param_t_excitation = 0.03

    use_mujoco_py = True  #whether to use mujoco-py or mujoco

    for trial_id, (username, table_filename) in enumerate(filelist):
        results_dir = f"_results/{username}_{CFAT_timestep}s/"
        if not os.path.exists(os.path.expanduser(results_dir)):
            os.makedirs(os.path.expanduser(results_dir), exist_ok=True)
        print(f'\nCOMPUTING FEASIBLE CONTROLS for {table_filename} with constant controls for {CFAT_timestep} seconds...')

        model_filename = f"models/OriginExperiment_{username}.xml"

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
                       timestep=CFAT_timestep,
                       results_dir=results_dir,
                       use_mujoco_py=use_mujoco_py)
