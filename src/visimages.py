# .box file format
# https://blake.bcm.edu/emanwiki/Eman2OtherFiles
# .box files
# 5341    8216    256     256     -3
# 3269    1876    256     256     -3
# 3386    2801    256     256     -3
# 2669    1080    256     256     -3
# There are 5 columns. The first 2 are the x/y coordinates of the lower left corner of the box. 
# The next 2 are the box size in X and Y (normally the same). 
# The fifth number is a 'mode', and isn't always present. 
# This value is normally used to identify filament boxes (with a starting and ending coordinate on paired lines)


import os
import argparse
import logging
import cv2
import numpy as np
import putils
from tqdm import tqdm

class Visualize(object):
    def __init__(self, image_path, annot_path, output_dir, annot_type, ddelay, factor):
        self.logger = logging.getLogger('PROTEIN')
        self.image_path = image_path
        self.annot_path = annot_path
        self.output_dir = output_dir
        self.annot_type = annot_type
        self.ddelay = ddelay
        self.factor = factor
        
    def validation(self):
        validation = True
        self.logger.info("Image Path: {}".format(self.image_path))
        if not os.path.exists(self.image_path):
            self.logger.critical("Input image path/directory doesn't exist")
            validation = False

        self.logger.info("Annotation Path: {}".format(self.annot_path))
        if not os.path.exists(self.annot_path):
            self.logger.critical("Annotation path/directory doesn't exist")
            validation = False
        
        self.logger.info("Output Dir: {}".format(self.output_dir))
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger.info("Annotation type: {}".format(self.annot_type))
        self.logger.info("Display Dealy: {}".format(self.ddelay))
        
        return validation

    def draw_bbox(self, image, annot_path, annot_type, factor):
        anf = open(annot_path, 'r')
        for line in anf:
            # if np.random.rand() < 0.85:
            #     continue

            if '\t' in line:
                line = line.strip().split('\t')
            else:
                line = line.strip().split()
                
            if len(line) < 4:
                self.logger.warning("Skipping annotation {}".format(line))
                continue
            xmin, ymin, xmax, ymax = putils.get_box_coordinates(line, image.shape[0], factor=factor)

            # Blue color in BGR
            color = (255, 0, 0)
            
            # Line thickness of 2 px
            thickness = 20
            # print(xmin, ymin, xmax, ymax)
            # Using cv2.rectangle() method
            # Draw a rectangle with blue line borders of thickness of 2 px
            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)

        anf.close()
        return image

    def _get_file_name(self, file_path):
        return os.path.basename(file_path)

    def get_filename_ext(self, file_name_ext):
        file_name, ext = os.path.splitext(file_name_ext)
        return file_name, ext

    def visualize(self):
        if not self.validation():
            self.logger.critical("Aborting execution. Please check input parameters")
            return

        if os.path.isfile(self.image_path):
            image = cv2.imread(self.image_path)
            image = self.draw_bbox(image, self.annot_path, self.annot_type)
            # Write image to disk
            cv2.imwrite(os.path.join(self.output_dir, self._get_file_name(self.image_path)), image)
            
            # Display the image
            cv2.namedWindow('Annotation', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Annotation', 600, 600)
            cv2.imshow('Annotation', image)
            key = cv2.waitKey(0)
            if key == 27:   # if ESC is pressed, exit loop
                cv2.destroyAllWindows()
            return 

        pbar = tqdm(sorted(os.listdir(self.image_path)))
        for image_name in pbar:
            pbar.set_description("Visualzing image: {}".format(image_name))
            # self.logger.info("Visualzing image: {}".format(image_name))
            file_name, ext = self.get_filename_ext(image_name)
            annot_path = os.path.join(self.annot_path, file_name + "." + self.annot_type)
            if not os.path.exists(annot_path):
                self.logger.critical("Annotation for image: {} doesn't exist, skipping the image".format(image_name))
                continue

            image = cv2.imread(os.path.join(self.image_path, image_name))
            image = self.draw_bbox(image, annot_path, self.annot_type, self.factor)
            # Write image to disk
            cv2.imwrite(os.path.join(self.output_dir, image_name), image)

        # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.equalizeHist(image)
        # alpha = 0.5 # Contrast control (1.0-3.0)
        # beta = 0 # Brightness control (0-100)
        # image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


if __name__ == '__main__':
    # Initialize logger
    from mylogger import init_logger
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    init_logger('PROTEIN', log_level)

    parser = argparse.ArgumentParser(description="Protein Object Detection")
    parser.add_argument('-i', '--image', action='store', dest='image_path', required=True,  help='Image path or directory hvaing images')
    parser.add_argument('-a', '--annotation', action='store', dest='annot_path', required=True, help='Annotation path or directory hvaing annotations')
    parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', required=True, help='Output direcotry')
    parser.add_argument('-t', '--type', action='store', dest='annot_type', default='box',  help='Annotation format')
    parser.add_argument('-d', '--delay', action='store', dest='ddelay', default=3000, type=int,  help='Image display delay in milliseconds')
    parser.add_argument('-f', '--factor', action='store', dest='factor', default=0.1, type=float,  help='BBox tight factor')

    args = parser.parse_args()
    vis = Visualize(args.image_path, args.annot_path, args.output_dir, args.annot_type, args.ddelay, args.factor)
    vis.visualize()


