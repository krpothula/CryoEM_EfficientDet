import os
import sys
import argparse
import logging
import pandas as pd
import yaml
import cv2
from tqdm import tqdm

# Initialize logger
from mylogger import init_logger
log_level = os.getenv('LOG_LEVEL', 'INFO')
init_logger('PROTEIN', log_level)

from data_preparation import DataPreperation
from visimages import Visualize
from coco_annotation import CocoAnnotation
import putils

sys.path.append("src/efficientdet/")
from efficientdet.dataset import create_coco_tfrecord

class Stage1(object):
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

    def cleanup_boxes(self):
        input_path = os.path.join(self.config['s1_base_dir'], self.config['s1_annotation_dir'])
        output_path = os.path.join(self.config['s1_base_dir'], self.config['s1_annot_cleanup'])
        os.makedirs(output_path, exist_ok=True)
        for infilename in sorted(os.listdir(input_path)):
            out_filename, ext = infilename.split('.')
            out_filename, _ = out_filename.rsplit('_', 1)
            out_filename = out_filename + '.' + ext
            anf = open(os.path.join(input_path, infilename), 'r')
            anf_out = open(os.path.join(output_path, out_filename), 'w')
            for line in anf:
                line = line.strip().split()
                line[2] = int(line[2])
                line[3] = int(line[3])
                line[0] = int(float(line[0])) - line[2] // 2
                line[1] = int(float(line[1])) - line[3] // 2
                anf_out.write("{}\t{}\t{}\t{}\n".format(line[0], line[1], line[2], line[3]))
            anf_out.close()
            anf.close()
            self.logger.info("Cleaned annotation: {}".format(out_filename))

    def split_tvt(self):
        data = pd.read_excel(os.path.join(self.config['s1_base_dir'], self.config['s1_datasplit']), 
                    sheet_name='Sheet1', usecols="A:C")
        input_path = os.path.join(self.config['s1_base_dir'], self.config['s1_image_dir'])
        train_path = os.path.join(self.config['s1_base_dir'], self.config['s1_train_data'])
        val_path = os.path.join(self.config['s1_base_dir'], self.config['s1_val_data'])
        test_path = os.path.join(self.config['s1_base_dir'], self.config['s1_test_data'])

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        self.logger.info("Copying images to train/validation/test direcotires")
        for _, row in tqdm(data.iterrows(), total=len(data)):
            # print('Processing the file:{}'.format(row['Image']), ' '*25, end='\r')
            image = cv2.imread(os.path.join(input_path, row['Image']))
            output_path = ''
            if row['Split'] == 'Train':
                output_path = os.path.join(train_path, row['Image'])
            elif row['Split'] == 'Validation':
                output_path = os.path.join(val_path, row['Image'])
            elif row['Split'] == 'Test':
                output_path = os.path.join(test_path, row['Image'])
            if output_path != '':
                cv2.imwrite(output_path, image)
        self.logger.info("Successfully copied images to train/validation/test directories")

    def data_prep(self):
        self.logger.info("Preperaring Train data")
        data_prep_train = DataPreperation(
            image_dir=os.path.join(self.config['s1_base_dir'], self.config['s1_train_data']),
            annot_dir=os.path.join(self.config['s1_base_dir'], self.config['s1_annot_cleanup']),
            annot_type='box',
            output_dir=os.path.join(self.config['s1_base_dir'], self.config['s1_train_img_path']),
            output_annot_dir=os.path.join(self.config['s1_base_dir'], self.config['s1_train_box_path']),
            out_dim=self.config['s1_dim'],
            partition=self.config['s1_partition'],
            testmode=False,
            factor=self.config['s1_factor'])
        data_prep_train.process()

        self.logger.info("Preperaring Validaiton data")
        data_prep_val = DataPreperation(
            image_dir=os.path.join(self.config['s1_base_dir'], self.config['s1_val_data']),
            annot_dir=os.path.join(self.config['s1_base_dir'], self.config['s1_annot_cleanup']),
            annot_type='box',
            output_dir=os.path.join(self.config['s1_base_dir'], self.config['s1_val_img_path']),
            output_annot_dir=os.path.join(self.config['s1_base_dir'], self.config['s1_val_box_path']),
            out_dim=self.config['s1_dim'],
            partition=self.config['s1_partition'],
            testmode=False,
            factor=self.config['s1_factor'])
        data_prep_val.process()

        self.logger.info("Preperaring Test data")
        data_prep_test = DataPreperation(
            image_dir=os.path.join(self.config['s1_base_dir'], self.config['s1_test_data']),
            annot_dir=os.path.join(self.config['s1_base_dir'], self.config['s1_annot_cleanup']),
            annot_type='box',
            output_dir=os.path.join(self.config['s1_base_dir'], self.config['s1_test_img_path']),
            output_annot_dir=None,
            out_dim=self.config['s1_dim'],
            partition=self.config['s1_partition'],
            testmode=True,
            factor=self.config['s1_factor'])
        data_prep_test.process()

    def visualize(self):
        self.logger.info("Visualizing Train images")
        vis_train = Visualize(
            image_path=os.path.join(self.config['s1_base_dir'], self.config['s1_train_img_path']),
            annot_path=os.path.join(self.config['s1_base_dir'], self.config['s1_train_box_path']),
            output_dir=os.path.join(self.config['s1_base_dir'], self.config['s1_vis_train']),
            annot_type='box', 
            ddelay=3000, 
            factor=0.0)
        vis_train.visualize()

        self.logger.info("Visualizing Validation images")
        vis_val = Visualize(
            image_path=os.path.join(self.config['s1_base_dir'], self.config['s1_val_img_path']),
            annot_path=os.path.join(self.config['s1_base_dir'], self.config['s1_val_box_path']),
            output_dir=os.path.join(self.config['s1_base_dir'], self.config['s1_vis_val']),
            annot_type='box', 
            ddelay=3000, 
            factor=0.0)
        vis_val.visualize()
        
    def gen_coco_annot(self):
        coco_train = CocoAnnotation(
            image_dir=os.path.join(self.config['s1_base_dir'], self.config['s1_train_img_path']), 
            annot_dir=os.path.join(self.config['s1_base_dir'], self.config['s1_train_box_path']), 
            output_path=os.path.join(self.config['s1_base_dir'], self.config['s1_train_coco']), 
            annot_type='box',
            img_width=self.config['s1_dim'],
            img_height=self.config['s1_dim'])
        coco_train.process()

        coco_val = CocoAnnotation(
            image_dir=os.path.join(self.config['s1_base_dir'], self.config['s1_val_img_path']), 
            annot_dir=os.path.join(self.config['s1_base_dir'], self.config['s1_val_box_path']), 
            output_path=os.path.join(self.config['s1_base_dir'], self.config['s1_val_coco']), 
            annot_type='box',
            img_width=self.config['s1_dim'],
            img_height=self.config['s1_dim'])
        coco_val.process()
    
    def create_tfrecord(self):
        tfrec_path = os.path.join(self.config['s1_base_dir'], self.config['s1_train_tfrecord'])
        os.makedirs(os.path.dirname(tfrec_path), exist_ok=True)
        create_coco_tfrecord._create_tf_record_from_coco_annotations(
            image_info_file=os.path.join(self.config['s1_base_dir'], self.config['s1_train_coco']),
                image_dir=os.path.join(self.config['s1_base_dir'], self.config['s1_train_img_path']),
                output_path=tfrec_path,
                num_shards=2,
                object_annotations_file=os.path.join(self.config['s1_base_dir'], self.config['s1_train_coco']),
                caption_annotations_file=None,
                include_masks=False, num_threads=1)

        tfrec_path = os.path.join(self.config['s1_base_dir'], self.config['s1_val_tfrecord'])
        os.makedirs(os.path.dirname(tfrec_path), exist_ok=True)
        create_coco_tfrecord._create_tf_record_from_coco_annotations(
            image_info_file=os.path.join(self.config['s1_base_dir'], self.config['s1_val_coco']),
                image_dir=os.path.join(self.config['s1_base_dir'], self.config['s1_val_img_path']),
                output_path=tfrec_path,
                num_shards=1,
                object_annotations_file=os.path.join(self.config['s1_base_dir'], self.config['s1_val_coco']),
                caption_annotations_file=None,
                include_masks=False, num_threads=1)

    def _convert_to_yolo_format(self,target_image, img_width, img_height, source_annotation, target_annotation):
        os.makedirs(target_annotation, exist_ok=True)
        for file_name in sorted(os.listdir(target_image)):
            file_name_base, ext = os.path.splitext(file_name)
            anf_in = os.path.join(source_annotation, file_name_base + '.box')
            anf_out = os.path.join(target_annotation, file_name_base + '.txt')
            anf_in_fp = open(anf_in, 'r')
            anf_out_fp = open(anf_out, 'w')
            for line in anf_in_fp:
                line = line.strip().split('\t')
                if len(line) < 4:
                    self.logger.warning("Skipping annotation {}".format(line))
                    continue
                xminb, yminb, xmaxb, ymaxb = putils.get_box_coordinates(line, img_height, factor=0.0)
                width = (xmaxb - xminb)
                height = (ymaxb - yminb)
                x_center = xminb +  width / 2
                y_center = yminb +  height / 2
                x_center /= img_width
                y_center /= img_height
                width /= img_width
                height /= img_height
                anf_out_fp.write("{} {} {} {} {}\n".format(0, x_center, y_center, width, height))
            anf_in_fp.close()
            anf_out_fp.close()

    def convert_to_yolo_format(self):
        self._convert_to_yolo_format(
            target_image=os.path.join(self.config['s1_base_dir'], self.config['s1_train_img_path']),
            img_width=self.config['s1_dim'],
            img_height=self.config['s1_dim'],
            source_annotation=os.path.join(self.config['s1_base_dir'], self.config['s1_train_box_path']),
            target_annotation=os.path.join(self.config['s1_base_dir'], self.config['s1_train_lbl_path']))
        self._convert_to_yolo_format(
            target_image=os.path.join(self.config['s1_base_dir'], self.config['s1_val_img_path']),
            img_width=self.config['s1_dim'],
            img_height=self.config['s1_dim'],
            source_annotation=os.path.join(self.config['s1_base_dir'], self.config['s1_val_box_path']),
            target_annotation=os.path.join(self.config['s1_base_dir'], self.config['s1_val_lbl_path']))
        
    def process(self):
        if not self.config['s1_start_step'] > 1:
            self.logger.info("Stage 1.1: Cleaning annotation box files")
            self.cleanup_boxes()

        if not self.config['s1_start_step'] > 2:
            self.logger.info("Stage 1.2: Split data into Train/Validation/Test")
            self.split_tvt()

        if not self.config['s1_start_step'] > 3:
            self.logger.info("Stage 1.3: Crop images into overlapping regions")
            self.data_prep()

        if not self.config['s1_start_step'] > 4:
            self.logger.info("Stage 1.4: Visualize annotation on training and validation images")
            self.visualize()
        
        if not self.config['s1_start_step'] > 5:
            self.logger.info("Stage 1.5: Convert annotations from box to coco format")
            self.gen_coco_annot()
        
        if not self.config['s1_start_step'] > 6:
            self.logger.info("Stage 1.6: Convert annotations from box to Yolo format")
            self.convert_to_yolo_format()
        
        if not self.config['s1_start_step'] > 7:
            self.logger.info("Stage 1.7: Create TensorflowRecords")
            self.create_tfrecord()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Protein Object Detection")
    parser.add_argument('-c', '--config', action='store', dest='config_path', required=True,  help='Path to configuration file')
    args = parser.parse_args()
    s1 = Stage1(args.config_path)
    s1.process()
