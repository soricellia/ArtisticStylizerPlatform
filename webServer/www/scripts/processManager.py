import os
import logging
import subprocess
import re
import pandas as pd

logger = logging.getLogger("processManager")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")

fileHandler = logging.FileHandler("processManager.log")
fileHandler.setFormatter(formatter)

logger.addHandler(fileHandler)

cw_dir = os.getcwd()
gpuServer_dir = os.path.join("/".join(cw_dir.split("/")[:-2]), "gpuServer/AS/src")

def get_tf_status(dir=gpuServer_dir, filename="adain_style_transfer.log"):
  logger.info("get_tf_status")
  last_line = subprocess.check_output(['tail', '-1', os.path.join(dir, filename)])
  pattern = "inference_master:end style transfer"
  if re.search(pattern, last_line):
    tf_free = True
  else:
    tf_free = False
  logger.info(last_line)
  logger.info(tf_free)
  return tf_free
# end

def load_json(file):
  logger.info("load_json")
  logger.info(file)
  #with open(file, "r") as json_file:
  #  persisted_processes = json.load(json_file)
  #logger.info(persisted_processes)
  persisted_processes = pd.read_json(file, orient="records")
  logger.info(persisted_processes)
  return persisted_processes
# end

def get_min_queue_rank_elm(df):
  logger.info("min_queue_rank")
  min_rank = min(df["queue_rank"].tolist())
  logger.info('min_rank:{}'.format(min_rank))
  return min_rank
# end

logger.info("############## next ############")
while True:
  tf_free = get_tf_status()
  logger.info("after checking tf status")
  if tf_free:
    logger.info("tf is free")
    queue_df = load_json(file="process_queue.json")
    min_rank = get_min_queue_rank_elm(df=queue_df)
  exit(0)
  #else:
    #wait
