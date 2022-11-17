import yaml
import os


def find_config_object(
    folder="dataset",
    name="dummy",
    key="datamodule",
    prefix="../../../../../../../../../../../private/home/meganrichards/projects/mise-en-place/config/",
):
    path = prefix + folder + "/" + name + ".yaml"
    print("\nLooking for: " + path)
    print("\nCWD: " + os.getcwd() + "\n")
    # print(os.path.abspath())
    # print(os.path.relpath())
    with open(path, "r") as stream:
        try:
            config_contents = yaml.safe_load(stream)
            print(config_contents[key])
            return config_contents[key]

        except yaml.YAMLError as exc:
            print(exc)
