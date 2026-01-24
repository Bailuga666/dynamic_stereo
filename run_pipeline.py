import os
import shutil
import subprocess

# Define the 4 input directories
INPUT_DIRS = [
    "/openbayes/home/sample_045/inputs",
    "/openbayes/home/sample_045/preds",
    "/openbayes/home/sample_050/inputs",
    "/openbayes/home/sample_050/preds",
    "/openbayes/home/sample_000/inputs",
    "/openbayes/home/sample_000/preds",
    "/openbayes/home/sample_001/inputs",
    "/openbayes/home/sample_001/preds",
]

# We need a temp directory structure for custom dataset
TEMP_BASE = "/openbayes/home/dynamic_stereo/temp_workspace"
OUTPUT_BASE = "/openbayes/home/dynamic_stereo/pipeline_outputs"

CHECKPOINT_PATH = "./checkpoints/dynamic_stereo_dr_sf.pth"

# Ensure script is run from dynamic_stereo root
os.chdir("/openbayes/home/dynamic_stereo")

for input_dir in INPUT_DIRS:
    print(f"Processing {input_dir}...")
    
    # Create a unique name for this run
    # /openbayes/home/sample_045/inputs -> openbayes_home_sample_045_inputs
    dir_name = input_dir.strip("/").replace("/", "_")
    temp_dir = os.path.join(TEMP_BASE, dir_name)
    output_dir = os.path.join(OUTPUT_BASE, dir_name)
    
    # Cleaning up prev run
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(os.path.join(temp_dir, "left"))
    os.makedirs(os.path.join(temp_dir, "right"))
    
    # Symlink images
    # Input files are like left_000.png, right_000.png
    files = sorted(os.listdir(input_dir))
    count = 0
    for f in files:
        src = os.path.join(input_dir, f)
        if f.startswith("left_") and f.endswith(".png"):
            dst = os.path.join(temp_dir, "left", f)
            os.symlink(src, dst)
            count += 1
        elif f.startswith("right_") and f.endswith(".png"):
            dst = os.path.join(temp_dir, "right", f)
            os.symlink(src, dst)
            count += 1
            
    print(f"Symlinked {count} images to {temp_dir}")
    
    # Run evaluation
    # Removed explicit model_weights argument as it matches default and was causing issues
    cmd = [
        "python", "evaluation/evaluate.py",
        f"exp_dir={output_dir}",
        "dataset_name=custom",
        f"data_root={temp_dir}",
        "--config-name", "eval_custom"
    ]
    
    print("Running evaluation...")
    subprocess.check_call(cmd)
    
    # Create visualization images and video
    disp_dir = os.path.join(output_dir, "disparities")
    vis_out_dir = os.path.join(output_dir, "visualization")
    
    if os.path.exists(disp_dir) and len(os.listdir(disp_dir)) > 0:
        cmd_vis = [
            "python", "generate_vis.py",
            "--disp_dir", disp_dir,
            "--out_dir", vis_out_dir,
            "--fps", "10",
            "--cmap", "viridis"
        ]
        
        print("Generating visualization images and video...")
        subprocess.check_call(cmd_vis)
        print(f"Finished {input_dir}. Results in {vis_out_dir}")
    else:
        print(f"No disparities found for {input_dir}")

print("All done!")
