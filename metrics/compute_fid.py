# metrics/compute_fid.py

import os
import subprocess
from pathlib import Path
import torch

def compute_fid(path1, path2, device="cuda"):
    """
    Compute FID score between two directories using pytorch-fid.

    Args:
        path1 (str): Path to the first image directory (e.g., reference images).
        path2 (str): Path to the second image directory (e.g., generated images).
        device (str, optional): Device to use ('cuda:0', 'cpu'). Defaults to None.
    
    Returns:
        float: The FID score.
    """
    # Ensure paths are strings
    path1 = str(path1)
    path2 = str(path2)
    
    
    # Build the command to compute FID
    cmd = [
        'python', '-m', 'pytorch_fid', path1, path2, '--device', device
    ]
    
    # Execute the command and capture the output
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Check if the command executed successfully
    if result.returncode != 0:
        print("Error computing FID:")
        print(result.stderr)
        raise RuntimeError("FID computation failed.")
    
    # Extract FID score from the output
    output = result.stdout.strip()
    try:
        fid_score = float(output.split()[-1])
    except ValueError:
        raise RuntimeError(f"Unexpected output from FID computation: {output}")
    
    return fid_score
