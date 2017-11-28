import os
import argparse
import json
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("content_img_path", type=str, help="full path to content image")
parser.add_argument("style_img_path", type=str, help="full path to style image")
parser.add_argument("result_img_path", type=str, help="full path to results directory where stylized content image is written")
parser.add_argument("final_size", default=256, type=int, help="final dimensions of the stylized content image")
parser.add_argument("transient_size", default=512, type=int, help="size images before cropping")
args = parser.parse_args()

current_process_data = {
        "content_img_path":args.content_img_path,
        "style_img_path":args.style_img_path,
        "result_img_path":args.result_img_path,
        "final_size":args.final_size,
        "transient_size":args.transient_size,
        "queue_rank":0
}

if os.path.exists(os.path.join(os.getcwd(), "process_queue.json")):
    with open("process_queue.json", "r") as json_file:
        persisted_processes = json.load(json_file)
else:
    os.mknod("process_queue.json")


#TODO get max element in queue
# todo update current element to max +1
# add current element to queue
# if there is no process running kick off next element in queue
