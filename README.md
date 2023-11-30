## Data formatting
`format_ie_input.py` contains options to format the inputs for both PubTator and OntoGPT.

## PubTator
This directory contains sample code (`pubtator_code`) from the [PubTator documentation](https://www.ncbi.nlm.nih.gov/research/pubtator/api.html) that can be used to annotate arbitrary text using the PubTator API, as well as code to format text into the required json format. The contents of the `input` and `output` directories are ignored, but the (mostly) empty directories are tracked so that they can be used by anyone cloning this repository.

## OntoGPT
The script `run_ontoGPT.sh` can be used to run OntoGPT over a set of text files in a directory. The directions to set up the dependencies are found in the [OntoGPT repo](https://github.com/monarch-initiative/ontogpt). In short, we cloned the repository, copied the `desiccation` schema from `information_extraction/schema` to `ontogpt/src/ontogpt/templates`, set up a conda environment with python 3.9 (separate from the one described in the main README of this repo) and used `pip install -e .` from the root directory of ontogpt to set up the package for our usage here.
