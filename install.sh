#!/bin/bash
# ________                  _______________                
# ___  __ \____________________  ____/__  /________      __
# __  / / /  _ \  _ \__  __ \_  /_   __  /_  __ \_ | /| / /
# _  /_/ //  __/  __/_  /_/ /  __/   _  / / /_/ /_ |/ |/ / 
# /_____/ \___/\___/_  .___//_/      /_/  \____/____/|__/  
#                  /_/                                    
# By Yazan Obeidi
# Inform user that installation has started
echo "Starting Deep Flow installation ..."
# Make project directory
mkdir ~/.deepflow
# Create virtual python environment
virtualenv -p python3 ~/.deepflow/deepflow 
# Add virtual python environment to jupyter notebook as a kernel
python3 -m ipykernel install --user --name deepflow --display-name "DeepFlow"
# Clone the git repository
git clone https://github.com/yazanobeidi/deep-flow ~/.deepflow
# Install python dependencies to virtual environment
~/.deepflow/deepflow/usr/bin/pip3 install -r ~/.deepflow/deep-flow/config/requirements.txt
# Create a bash alias so user can simply run "$ deepflow"
echo "alias deepflow=\"~/.deepflow/deep-flow/start.sh\"" >> ~/.bash_aliases
# Inform user that installation has successfully completed
echo "Deep Flow Installation Complete!"
# Create default data directory
mkdir ~/.deepflow/deep-flow/data
# Prompt user to add dataset to default data directory or update config
echo "To begin, you may add your csv files to ~/.deepflow/deep-flow/data or edit config/config.cfg to point to your dataset"