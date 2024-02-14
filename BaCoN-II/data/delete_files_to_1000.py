import os
import sys  # Import sys to access command-line arguments

def delete_files_in_directory(directory):
    # Get all files in the specified directory
    files = os.listdir(directory)

    # Loop through each file
    for file in files:
        # Extract the filename without extension
        filename, extension = os.path.splitext(file)
        
        # Check if the filename is numeric and greater than 1000
        if filename.isdigit() and int(filename) > 1000:
            # Construct the full path to the file
            file_path = os.path.join(directory, file)
            # Delete the file
            os.remove(file_path)
           

# Loop through each directory specified in the command line
for directory_path in sys.argv[1:]:  # Skip the first argument, which is the script name
    print(f"Processing directory: {directory_path}")
    delete_files_in_directory(directory_path)