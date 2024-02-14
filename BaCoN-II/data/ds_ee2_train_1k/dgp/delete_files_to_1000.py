import os

# Get all files in the current directory
files = os.listdir('.')

# Loop through each file
for file in files:
    # Extract the filename without extension
    filename, extension = os.path.splitext(file)
    
    # Check if the filename is numeric and greater than 1000
    if filename.isdigit() and int(filename) > 1000:
        # Construct the full path to the file
        file_path = os.path.join('.', file)
        # Delete the file
        os.remove(file_path)
