# Specify the file path
file_path = "/srv/upadro/HiAGM/data/rcv1_test.json"

# Read the file
with open(file_path, 'r') as file:
    content = file.read()

# Perform the replacement
modified_content = content.replace("Label", "label")

# Write the modified content back to the file
with open(file_path, 'w') as file:
    file.write(modified_content)

