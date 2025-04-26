import os
import numpy as np

def convert_npy_to_float32(input_folder):
    """
    Convert all .npy files in a folder and its subfolders to float32 
    and overwrite the original files.
    
    Parameters:
        input_folder (str): Path to the folder containing .npy files.
    """
    # Folders to skip
    skip_folders = {"sub-11", "sub-06", "avg_rdms_sbjct_avg", "raw_rdms_sbjct_avg", "networks_rdms"}

    for root, dirs, files in os.walk(input_folder):
        # Remove skipped folders from the dirs list
        dirs[:] = [d for d in dirs if d not in skip_folders]

        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(root, file)
                
                try:
                    # Load the .npy file
                    data = np.load(file_path)
                    
                    # Convert to float32 if not already
                    if data.dtype != np.float32:
                        print(f"Converting {file_path} from {data.dtype} to float32.")
                        data = data.astype(np.float32)
                        
                        # Save back to the same file
                        np.save(file_path, data)
                    else:
                        print(f"{file_path} is already float32, skipping.")
                
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Example usage
path = "/path/to/your/folder"  # Replace with your folder path
convert_npy_to_float32(path)

