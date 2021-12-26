import os
import argparse
import logging
import json
import cv2
import numpy as np
import pandas as pd
import putils

class CocoAnnotation(object):
    def __init__(self, image_dir, annot_dir, output_path, annot_type, img_width, img_height):
        self.logger = logging.getLogger('PROTEIN')
        self.image_dir = image_dir
        self.annot_dir = annot_dir
        self.output_path = output_path
        self.annot_type = annot_type
        self.img_width = img_width
        self.img_height = img_height

    def validation(self):
        validation = True
        self.logger.info("Image Dir: {}".format(self.image_dir))
        if not os.path.exists(self.image_dir):
            self.logger.critical("Input image directory doesn't exist")
            validation = False

        self.logger.info("Annotation Dir: {}".format(self.annot_dir))
        if not os.path.exists(self.annot_dir):
            self.logger.critical("Annotation directory doesn't exist")
            validation = False
        
        self.logger.info("Output Path: {}".format(self.output_path))
        if os.path.exists(self.output_path):
            self.logger.warning("Output file already exist, will be overwritten")
        
        return validation

    def process(self):
        if not self.validation():
            self.logger.critical("Aborting execution. Please check input parameters")
            return

        json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
        json_dict['categories'].append({
            'id': 1,
            'name': 'protein',
            'supercategory': 'protein', 
        })
        
        annot_id = 0
        for image_id, image_name in enumerate(sorted(os.listdir(self.image_dir))):
            self.logger.debug("Converting to coco annotation image: {}".format(image_name))
            file_name, ext = putils.get_filename_ext(image_name)
            annot_path = os.path.join(self.annot_dir, file_name + "." + self.annot_type)
            if not os.path.exists(annot_path):
                self.logger.critical("Annotation for image: {} doesn't exist, skipping the image".format(image_name))
                continue
            # image = cv2.imread(os.path.join(self.image_dir, image_name))
            # h, w, _ = image.shape
            h, w = self.img_height, self.img_width
            json_dict['images'].append({
                'file_name': image_name, 
                'height': h, 
                'width': w, 
                'id': image_id
            })
            anf = open(annot_path, 'r')
            for line in anf:
                line = line.strip().split()
                if len(line) < 4:
                    self.logger.warning("Skipping annotation {}".format(line))
                    continue
                xmin, ymin, xmax, ymax = putils.get_box_coordinates(line, self.img_width, factor=0.0)
                b_width = xmax - xmin
                b_height = ymax - ymin

                json_dict['annotations'].append({
                    'area': b_width * b_height,
                    'iscrowd': 0,
                    'image_id': image_id,
                    'bbox': [xmin, ymin, b_width, b_height],
                    'category_id': 1,
                    'id': annot_id,
                    'segmentation': []

                })
                annot_id += 1
            anf.close()

        json_fp = open(self.output_path, 'w', encoding='utf-8')
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
        json_fp.close()


if __name__ == '__main__':
    # Initialize logger
    from mylogger import init_logger
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    init_logger('PROTEIN', log_level)

    parser = argparse.ArgumentParser(description="Protein Object Detection")
    parser.add_argument('-i', '--image_dir', action='store', dest='image_dir', required=True,  help='Direcotry containing images')
    parser.add_argument('-a', '--annotation_dir', action='store', dest='annot_dir', required=True, help='Direcotry containing annotation')
    parser.add_argument('-o', '--output_path', action='store', dest='output_path', required=True, help='Output json path')
    parser.add_argument('-t', '--type', action='store', dest='annot_type', default='box',  help='Annotation format')
    parser.add_argument('--img_width', action='store', dest='img_width', type=int, default=768, help='Image width')
    parser.add_argument('--img_height', action='store', dest='img_height', type=int, default=768, help='Image height')

    args = parser.parse_args()
    cocoannot = CocoAnnotation(args.image_dir, args.annot_dir, args.output_path, args.annot_type, args.img_width, args.img_height)
    cocoannot.process()
    


