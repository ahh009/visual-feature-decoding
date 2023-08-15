from cloudpathlib import S3Path, S3Client
from utils import ls, crawl
import nibabel as nib
from pathlib import Path
import os
import s3fs
from utils import load_aws_credentials
from shutil import rmtree
import json
import pandas as pd


def read_json(path):
    with open(path) as f:
        return json.load(f)

def file_load(path, sub_name, client):
    path = S3Path(str(path).replace("{SUB}", sub_name), client=client)
    ext = Path(path).suffix
    extentions_d = {
                    ".gz": lambda x: nib.load(x),
                    ".csv": lambda x: pd.read_csv(x),
                    ".trk": lambda x: nib.streamlines.load(x),
                    ".json": read_json
                    }
    return extentions_d[ext](path)

def get_HCP_sub(file_paths, data_type="preprocessed"):
    """
    This function will take a list of file paths and return a generator that yields a dictionary of the file paths
    for each subject in the HCP dataset and the files for that subject.
    the files will be downloaded to "/temp/cache" and deleted after the generator yields the data.
    :param file_paths: a list of the paths to the files to be downloaded, the path should include "{SUB}" where the
    subject number should be in the file name. include the part of the path that is the same for all subjects (that is,
    everything after the subject folder).
    :param data_type: either preprocessed or diffusion, to indicate which dataset you want to download from
    :return: subject name and dictionary of file paths and the files
    """
    cache_path = Path('/tmp/cache')
    if os.path.isdir(cache_path):
        rmtree(cache_path)
    os.mkdir(cache_path)

    # Create a client that uses our cache path and that does not try to
    # authenticate with S3.
    client = S3Client(
        local_cache_dir=cache_path,
        no_sign_request=True)

    if data_type == "preprocessed":
        hcp_derivs_path = S3Path("s3://hcp-openaccess/HCP_1200", client=client)
    elif data_type == "diffusion":
        hcp_derivs_path = S3Path(
            "s3://open-neurodata/rokem/hcp1200/afq",
            client=client)
    else:
        raise TypeError("data_type should be 'preprocessed' or 'diffusion'")

    for sub in hcp_derivs_path.glob("*"):
        if os.path.isdir(cache_path):
            rmtree(cache_path)
        os.mkdir(cache_path)
        sub_name = os.path.split(sub)[-1]
        data = {file_path: file_load(sub / file_path, sub_name, client) for file_path in file_paths}
        yield sub_name, data

# usage example:
for sub, files in get_HCP_sub(["ses-01/{SUB}_dwi_space-RASMM_model-CSD_desc-prob-afq-clean_tractography.trk"], "diffusion"):
    tract_file = files["ses-01/{SUB}_dwi_space-RASMM_model-CSD_desc-prob-afq-clean_tractography.trk"]
    get_connectivity_matrix(tract_file)