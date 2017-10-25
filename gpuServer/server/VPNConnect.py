import os
import subprocess
import time
import re
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument("-st", "--sleep_time", type=int, default=5)

args = parser.parse_args()

# log file
vpnConnectionLogFile = "vpnConnection.log"

# text file containing vpn status results
confirmationFile = "vpnstatusResults.txt"
targetConfirmationPhrase = "You\sare\sconnected\svia\sthe\sCampus\sVPN"

# vpn connection commands
connectCMD = "printf '2514736egc\n1\nkhrqh\nmoneymo' | sudo openconnect v1.marist.edu"
confirmConnectCMD = "rm {}; w3m -dump http://www.marist.edu/it/network/vpnstatus.html > {}".format(confirmationFile, confirmationFile)

# set up logging
#logger = logging.getLogger(__name__)
logger = logging.getLogger("VPNConnection")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(vpnConnectionLogFile)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# ddd the handlers to the logger
logger.addHandler(handler)


# vpn connection command
def connect():
    logger.info("started VPNConnection")
    try:
        p = subprocess.Popen(connectCMD, shell=True, stdout=subprocess.PIPE)
        out, err = p.communicate()
    except Exception as e:
        logger.info("error with vpn connection") 
# end

# start initial connection
connect()

# check connection status
checkConnection = True
while checkConnection:
    # check status command
    p = subprocess.Popen(confirmConnectCMD, shell=True, stdout=subprocess.PIPE).wait()
    out, err = p.communicate()

    # check connection status
    try:
        with open(confirmationFile, "r") as file:
            content = file.read()
            match = re.search(targetConfirmationPhrase, content)
            file.close()
    except Exception as e:
        logger.info("error reading {}".format(vpnConnectionLogFile))
    
    # if not connected to vpn then connect to vpn
    if match is None:
        logger.info("connection lost")
        # vpn connection command
        connect()
    else:
        logger.info("connection maintained")
        time.sleep(args.sleep_time)
