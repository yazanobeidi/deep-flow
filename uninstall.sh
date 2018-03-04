#!/bin/bash
# ________                  _______________                
# ___  __ \____________________  ____/__  /________      __
# __  / / /  _ \  _ \__  __ \_  /_   __  /_  __ \_ | /| / /
# _  /_/ //  __/  __/_  /_/ /  __/   _  / / /_/ /_ |/ |/ / 
# /_____/ \___/\___/_  .___//_/      /_/  \____/____/|__/  
#                  /_/                                    
# By Yazan Obeidi
# Inform user that installation has started
echo "Removing Deep Flow ..."
# Delete project directory
rm ~/.deepflow
# Remove the bash alias
sed '/alias deepflow="~\/.deepflow\/deep-flow\/start.sh"/d' ~/.bash_aliases > ~/.bash_aliases
# Inform user that uninstallation has successfully completed
echo "Deep Flow Uninstalled."