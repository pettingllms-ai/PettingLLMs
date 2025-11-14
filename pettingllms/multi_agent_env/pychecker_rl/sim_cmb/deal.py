# Move testbench.json files from output_tb_gen_tb_20250407 directory to corresponding taskid folders in testcase directory
import os
import shutil

# Get current directory
current_dir = os.getcwd()

# Get all files in output_tb_gen_tb_20250407 directory
output_dir = os.path.join(current_dir, "output_tb_gen_tb_20250407")
output_files = os.listdir(output_dir)

# Get all files in testcase directory
testcase_dir = os.path.join(current_dir, "testcase")
testcase_files = os.listdir(testcase_dir)

# Iterate through all files in output_tb_gen_tb_20250407 directory
for file in output_files:
    # Get filename
    filename = os.path.basename(file)
    
    # Get taskid from filename
    taskid = filename.split("_")[0]
    
    # Find corresponding taskid file in testcase directory
    for testcase_file in testcase_files:
        if taskid in testcase_file:
            # If found, move file to testcase directory
            shutil.move(os.path.join(output_dir, file), os.path.join(testcase_dir, testcase_file))
            break
