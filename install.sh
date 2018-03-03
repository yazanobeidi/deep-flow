echo "Starting Deep Flow installation ..."
mkdir ~/.deepflow
cd ~/.deepflow
virtualenv -p python3 ~/.deepflow/deepflow 
python3 -m ipykernel install --user --name deepflow --display-name "DeepFlow"
git clone https://github.com/yazanobeidi/deep-flow
echo "alias deepflow=\"~/.deepflow/deep-flow/start.sh\"" >> ~/.bash_aliases
echo "Deep Flow Installation Complete!"