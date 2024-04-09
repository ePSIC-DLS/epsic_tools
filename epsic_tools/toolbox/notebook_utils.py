import json
import logging
import re
import os
import nbformat
import yaml


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class NotebookHelper:
    """
    This workflow takes a notebook. 
    """

    def __init__(self, notebook_paths, notebook_name):
        self.__notebook_paths = notebook_paths
        self.__notebook_name = notebook_name

    # ------------------------------------------------------------------
    # Method to get settings from notebook cell.
    def get_settings(self, cell_index):
        """
        Get settings from a notebook cell. 
        """

        ipynb_filename = os.path.join(self.__notebook_paths , f"{self.__notebook_name}.ipynb")

        logger.debug("importing jupyter code")

        # Read the notebook into memory.
        logger.debug(f"reading notebook from {ipynb_filename}")
        with open(ipynb_filename, "r") as stream:
            notebook = nbformat.read(stream, as_version=4)

        source = notebook["cells"][cell_index]["source"].strip()

        if len(source) == 0:
            raise RuntimeError(
                f"notebook {self.__notebook_name} cell {cell_index} is blank"
            )

        # Replace some markdown things that might be in there.
        source = re.sub(r"^(```yaml)(\n)?", "", source)
        source = re.sub(r"^(```json)(\n)?", "", source)
        source = re.sub(r"^(```)(\n)?", "", source)
        source = re.sub(r"(\n)?(```)$", "", source)

        if source[0] == "{":
            try:
                settings_dicts = json.loads(source)
            except Exception:
                raise RuntimeError(
                    f"notebook {self.__notebook_name} cell {cell_index} failed to parse as json"
                )
        else:
            try:
                settings_dicts = yaml.safe_load(source)
            except Exception:
                raise RuntimeError(
                    f"notebook {self.__notebook_name} cell {cell_index} failed to parse as yaml"
                )

        return settings_dicts


    def set_settings(self, new_settings, save_path, blank_cell_index=2):
        """
        Set settings to new values and saves a new version of notebook.
        """
#TODO: here the loading of notebook is repeated - maybe have it as a separate function?
        ipynb_filename = os.path.join(self.__notebook_paths , f"{self.__notebook_name}.ipynb")

        logger.debug("importing jupyter code")

        # Read the notebook into memory.
        logger.debug(f"reading notebook from {ipynb_filename}")
        with open(ipynb_filename, "r") as stream:
            notebook = nbformat.read(stream, as_version=4)
#TODO: Check here not to overwrite the dictionary

        source = []
        for key, value in new_settings.items():
            if type(value) is dict:
                source.append(f"{key}='{value['value']}'\n")
            else:
                source.append(f"{key}='{value}'\n") 
        source = ''.join(source)
        notebook["cells"][blank_cell_index]["source"] = source
        nbformat.write(notebook, save_path)
        return
#TODO: Bring submit option / code here


