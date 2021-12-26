# https://github.com/ultralytics/yolov5/issues/1605
import argparse
import os
import logging
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import glob
import torch

class YoloPredict(object):
    def __init__(self, saved_model_dir, input_image, model_name, min_score_thresh, 
                    output_prediction, output_image_dir):
        self.logger = logging.getLogger('PROTEIN')
        self.saved_model_dir = saved_model_dir
        self.input_image = input_image
        self.model_name = model_name
        self.min_score_thresh = min_score_thresh
        self.output_prediction = output_prediction
        self.output_image_dir = output_image_dir

        if self.output_image_dir is not None:
            self.print_output = True
            os.makedirs(self.output_image_dir, exist_ok=True)
        else:
            self.print_output = False

        self.model = torch.hub.load('src/yolov5/', 'custom', path=self.saved_model_dir, source='local')
        self.model.conf = self.min_score_thresh
        self.logger.info("Loaded model")
    
    def process(self):
        all_files = list(glob.glob(self.input_image))
        print("Number of images to process: {}".format(len(all_files)))
        pbar = tqdm(sorted(all_files))
        for image_name in pbar:
            pbar.set_description(image_name)
            file_name = os.path.basename(image_name)
            results = self.model(image_name)
            prediction = results.pandas().xyxy[0].copy()
            prediction['FileName'] = file_name
            prediction[['xmin', 'ymin', 'xmax', 'ymax']] = prediction[['xmin', 'ymin', 'xmax', 'ymax']].round()
            prediction.rename({'confidence':'score'}, axis='columns', inplace=True)
            prediction = prediction[['FileName', 'class', 'score', 'xmin', 'ymin', 'xmax', 'ymax']]
            with open(self.output_prediction, 'a') as f:
                prediction.to_csv(f, mode='a', header=f.tell()==0, index=False, line_terminator='\n')
            
            
            if self.print_output:
                results.save(self.output_image_dir)
                

        


