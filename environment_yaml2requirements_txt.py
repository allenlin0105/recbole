import os
import yaml

with open("environment.yml") as file_handle:
    environment_data = yaml.safe_load(file_handle)
    # print(environment_data["dependencies"][-1]["pip"])

with open("requirements.txt", "w") as fp:
    for lib in environment_data["dependencies"][-1]["pip"]:
        fp.write(lib + "\n")
