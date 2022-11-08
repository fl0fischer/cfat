import os
from zipfile import ZipFile
from io import BytesIO
from urllib.request import urlopen
import cfat

def check_study_dataset_IK_dir(DIRNAME_STUDY_IK):
    if not os.path.exists(DIRNAME_STUDY_IK):
        download_datasets = input(
            "Could not find reference to the raw IK data included in the ISO-VR-Pointing Dataset. Do you want to download it (~130MB after unpacking)? (y/N) ")
        if download_datasets.lower().startswith("y"):
            print(f"Will download and unzip to '{DIRNAME_STUDY_IK}'.")
            print("Downloading archive... ", end='', flush=True)
            resp = urlopen("https://zenodo.org/record/7300062/files/ISO_VR_Pointing_IK_Raw.zip?download=1")
            zipfile = ZipFile(BytesIO(resp.read()))
            print("unzip archive... ", end='', flush=True)
            for file in zipfile.namelist():
                zipfile.extract(file, path=DIRNAME_STUDY_IK)
            print("done.")
            assert os.path.exists(DIRNAME_STUDY_IK), "Internal Error during unpacking of ISO-VR-Pointing Dataset."
        else:
            raise FileNotFoundError("Please ensure that 'DIRNAME_STUDY' points to a valid directory containing the ISO-VR-Pointing Dataset.")


DIRNAME_STUDY_IK = "study_IK_raw/"
check_study_dataset_IK_dir(DIRNAME_STUDY_IK)

print('\n\n                              +++ FEASIBLE CONTROL COMPUTATION +++')
filelist = [(username, os.path.abspath(os.path.join(DIRNAME_STUDY_IK, f))) for username in [f"U{i}" for i in range(1, 7)]
            for f in os.listdir(DIRNAME_STUDY_IK) if
            os.path.isfile(os.path.abspath(os.path.join(DIRNAME_STUDY_IK, f))) and
            f.startswith(username) and f.endswith('.mot')]

job_directory = f"{os.getcwd()}/_slurm/jobs"
if not os.path.exists(job_directory):
    os.makedirs(job_directory, exist_ok=True)
output_directory = f"{os.getcwd()}/_slurm/out"
if not os.path.exists(output_directory):
    os.makedirs(output_directory, exist_ok=True)
error_directory = f"{os.getcwd()}/_slurm/err"
if not os.path.exists(error_directory):
    os.makedirs(error_directory, exist_ok=True)

use_mujoco_py = True  #whether to use mujoco-py or mujoco

for trial_id, (username, table_filename) in enumerate(filelist):
    print(f'\nCOMPUTING FEASIBLE CONTROLS for {table_filename}.')

    job_file = os.path.join(job_directory, f"{trial_id}.job")
    with open(job_file, 'w+') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f"#SBATCH -J {table_filename.split('/')[-1]}\n")
        fh.writelines(f"#SBATCH -N 1 # (Number of requested nodes)\n")
        #fh.writelines(f"#SBATCH --ntasks-per-node 20 # (Number of requested cores per node)\n")
        fh.writelines(f"#SBATCH -t 24:00:00 # (Requested wall time)\n")
        fh.writelines(f"#SBATCH --output={output_directory}/{trial_id}.out\n")
        fh.writelines(f"#SBATCH --error={error_directory}/{trial_id}.err\n")
        fh.writelines(f"srun python iso_vr_pointing_example.py --username={username} --table_filename={table_filename} {'--mujoco-py' if use_mujoco_py else '--mujoco'}")

    os.system(f"sbatch {job_file}")
