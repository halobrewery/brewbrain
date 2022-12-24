import os

def find_file_cwd_and_parent_dirs(file_name, start_dir):
  curr_dir = start_dir
  # For now we're only searching the current directory and parent directories
  while True:
    file_list = os.listdir(curr_dir)
    parent_dir = os.path.dirname(curr_dir)
    if file_name in file_list:
      return os.path.join(curr_dir, file_name)
    else:
      if curr_dir == parent_dir:
        # No file found in parents
        break
      else:
        curr_dir = parent_dir
        