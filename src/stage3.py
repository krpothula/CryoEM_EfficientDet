import os
import sys
import argparse
import logging
import yaml

# Initialize logger
from mylogger import init_logger
log_level = os.getenv('LOG_LEVEL', 'INFO')
init_logger('PROTEIN', log_level)

# import tensorflow.compat.v1 as tf
sys.path.append("src/efficientdet/")
# sys.path.append("src/yolov5/")
from visimages import Visualize
from predict import Predict
from postprocess import Postprocess
from yolopredict import YoloPredict

class Stage3(object):
    def __init__(self, config_path):
        self.logger = logging.getLogger('PROTEIN')
        self.config_path = config_path
        self.load_config()

    def load_config(self):
        self.logger.info("Loading config file from: {}".format(self.config_path))
        with open(self.config_path, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    
    def do_prediction(self):
        output_image_dir = None
        if self.config['s3_edd0_output_image_dir'] is not None:
            output_image_dir = os.path.join(self.config['s1_base_dir'], self.config['s3_edd0_output_image_dir'])
        pred = Predict(
            saved_model_dir=os.path.join(self.config['s1_base_dir'], self.config['s3_edd0_saved_model_dir']), 
            input_image=os.path.join(self.config['s1_base_dir'], self.config['s3_test_img_path']), 
            model_name='efficientdet-d0', 
            min_score_thresh=0.5,
            max_boxes_to_draw=150, 
            batch_size=1, 
            output_prediction=os.path.join(self.config['s1_base_dir'], self.config['s3_edd0_output_prediction_raw']), 
            hparams=self.config['s3_edd0_config'], 
            output_image_dir=output_image_dir)
        pred.process()
        
    def do_postprocess(self, model='edd0'):
        if model == 'edd0':
            output_prediction=os.path.join(self.config['s1_base_dir'], self.config['s3_edd0_output_prediction_raw'])
            final_prediction=os.path.join(self.config['s1_base_dir'], self.config['s3_edd0_output_prediction_final'])
            final_annotation_dir=os.path.join(self.config['s1_base_dir'], self.config['s3_edd0_output_prediction_box'])
        else:
            output_prediction=os.path.join(self.config['s1_base_dir'], self.config['s3_yv5s_output_prediction_raw'])
            final_prediction=os.path.join(self.config['s1_base_dir'], self.config['s3_yv5s_output_prediction_final'])
            final_annotation_dir=os.path.join(self.config['s1_base_dir'], self.config['s3_yv5s_output_prediction_box'])
            
        pp = Postprocess(
                output_prediction=output_prediction,
                final_prediction=final_prediction,
                final_annotation_dir=final_annotation_dir,
                annot_type='box', 
                out_dim=self.config['s1_dim'],
                partition=self.config['s1_partition'],
                image_dim=self.config['s1_image_dim'])
        pp.process()
    
    def do_yv5s_predict(self):
        output_image_dir = None
        if self.config['s3_yv5s_output_image_dir'] is not None:
            output_image_dir = os.path.join(self.config['s1_base_dir'], self.config['s3_yv5s_output_image_dir'])

        yp = YoloPredict(
            saved_model_dir=os.path.join(self.config['s1_base_dir'], self.config['s3_yv5s_saved_model_dir']),
            input_image=os.path.join(self.config['s1_base_dir'], self.config['s3_test_img_path']),
            model_name = 'yolov5s',
            min_score_thresh = 0.5, 
            output_prediction=os.path.join(self.config['s1_base_dir'], self.config['s3_yv5s_output_prediction_raw']),
            output_image_dir=output_image_dir)
        yp.process()
    
    def visualize(self, model='edd0'):
        self.logger.info("Visualizing Test final images")
        if model == 'edd0':
            annot_path=os.path.join(self.config['s1_base_dir'], self.config['s3_edd0_output_prediction_box'])
            output_dir=os.path.join(self.config['s1_base_dir'], self.config['s3_edd0_output_image_dir_final'])
        else:
            annot_path=os.path.join(self.config['s1_base_dir'], self.config['s3_yv5s_output_prediction_box'])
            output_dir=os.path.join(self.config['s1_base_dir'], self.config['s3_yv5s_output_image_dir_final'])

        vis_test = Visualize(
            image_path=os.path.join(self.config['s1_base_dir'], self.config['s1_test_data']),
            annot_path=annot_path,
            output_dir=output_dir,
            annot_type='box', 
            ddelay=3000, 
            factor=0.0)
        vis_test.visualize()
    
    def process(self):
        if self.config['s3_edd0_process']:
            self.logger.info("Processing EfficientDet model")
            if not self.config['s3_start_step'] > 1:
                self.logger.info("Stage 3.1: Prediction efficientdet")
                self.do_prediction()

            if not self.config['s3_start_step'] > 2:
                self.logger.info("Stage 3.2: Post processing")
                self.do_postprocess()
            
            if not self.config['s3_start_step'] > 3:
                self.logger.info("Stage 3.3: Visualize prediction")
                self.visualize()

        if self.config['s3_yv5s_process']:
            self.logger.info("Processing YoloV5 model")
            if not self.config['s3_start_step'] > 1:
                self.logger.info("Stage 3.1: Prediction Yolov5 model")
                self.do_yv5s_predict()

            if not self.config['s3_start_step'] > 2:
                self.logger.info("Stage 3.2: Post processing")
                self.do_postprocess(model='yv5s')
            
            if not self.config['s3_start_step'] > 3:
                self.logger.info("Stage 3.3: Visualize prediction")
                self.visualize(model='yv5s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Protein Object Detection")
    parser.add_argument('-c', '--config', action='store', dest='config_path', required=True,  help='Path to configuration file')
    args = parser.parse_args()
    s3 = Stage3(args.config_path)
    s3.process()
