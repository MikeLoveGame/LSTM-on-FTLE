import os
import shutil

source_directory = r"C:\AI\Data\CT-images\archive\images\images"  # Replace with the path to the source directory
destination_directory = r"C:\AI\CNIC\SAM\Data"  # Replace with the path to the destination directory

search_id = "ID00007637202177411956430"
def grab_files(source_directory, destination_directory, search_id):
    for file_name in os.listdir(source_directory):
        # Check if the file name contains the search ID
        if search_id in file_name:
            # Construct the destination file path
            destination_path = os.path.join(destination_directory, file_name)
            file_path= os.path.join(source_directory, file_name)
            # Copy the file to the destination directory
            shutil.copy(file_path, destination_path)

            # Optional: Print the copied file path
            print(f"Copied file: {file_path} to {destination_path}")


grab_files(source_directory, destination_directory, search_id)