from genericpath import exists
import os
import argparse
import logging
import cv2
from tqdm import tqdm
import putils


class DataPreperation(object):
    def __init__(self, image_dir, annot_dir, annot_type, output_dir, output_annot_dir, out_dim, 
                    partition, testmode, factor):
        self.logger = logging.getLogger('PROTEIN')
        self.image_dir = image_dir
        self.annot_dir = annot_dir
        self.annot_type = annot_type
        self.output_dir = output_dir
        self.output_annot_dir = output_annot_dir
        self.out_dim = out_dim
        self.partition = partition
        self.testmode = testmode
        self.factor = factor

    def validation(self):
        validation = True
        self.logger.info("Image Dir: {}".format(self.image_dir))
        if not os.path.exists(self.image_dir):
            self.logger.critical("Input image directory doesn't exist")
            validation = False

        self.logger.info("Output Dir: {}".format(self.output_dir))
        os.makedirs(self.output_dir, exist_ok=True)

        if not self.testmode:
            self.logger.info("Annotation Dir: {}".format(self.annot_dir))
            if not os.path.exists(self.annot_dir):
                self.logger.critical("Annotation directory doesn't exist")
                validation = False
        
            self.logger.info("Output Annotation Dir: {}".format(self.output_annot_dir))
            os.makedirs(self.output_annot_dir, exist_ok=True)

        self.logger.info("Output image dimension: {} x {}".format(self.out_dim, self.out_dim))
        self.logger.info("Image is split into {} x {} partitions".format(self.partition, self.partition))
        return validation

    def get_annot_list(self, file_name, image, xmin, ymin, xmax, ymax):
        anf_out_list = []
        annot_path = os.path.join(self.annot_dir, file_name + '.' + self.annot_type)
        anf = open(annot_path, 'r')
        for line in anf:
            line = line.strip().split('\t')
            if len(line) < 4:
                self.logger.warning("Skipping annotation {}".format(line))
                continue
            xminb, yminb, xmaxb, ymaxb = putils.get_box_coordinates(line, image.shape[0], factor=self.factor)
            overlap = putils.get_overlap((xmin, ymin, xmax, ymax), (xminb, yminb, xmaxb, ymaxb))
            if overlap > 0.75:
                width = xmaxb - xminb
                height = ymaxb - yminb
                xmin_new = xminb - xmin
                if xmin_new < 0:
                    width -= xmin_new
                    xmin_new = 0
                if xmin_new + width > self.out_dim:
                    width -= (xmin_new + width - self.out_dim)

                ymax_new = ymaxb - ymin
                if ymax_new > self.out_dim:
                    height -= (ymax_new - self.out_dim)
                    ymax_new = self.out_dim
                if yminb < ymin:
                    height -= (ymin - yminb)

                # Invert the ymax measurement
                ymax_new = self.out_dim - ymax_new
                anf_out_list.append([xmin_new, ymax_new, width, height])
        anf.close()
        return anf_out_list
        
    def process(self):
        if not self.validation():
            self.logger.critical("Aborting execution. Please check input parameters")
            return
        pbar = tqdm(sorted(os.listdir(self.image_dir)))
        for image_name in pbar:
            pbar.set_description("Cropping image: {}".format(image_name))
            # self.logger.info("Cropping image: {}".format(image_name))
            file_name, ext = putils.get_filename_ext(image_name)

            # If annotation doesn't exist skip it
            # Perform this check only for train and validation data
            if not self.testmode:
                annot_path = os.path.join(self.annot_dir, file_name + '.' + self.annot_type)
                if not os.path.exists(annot_path):
                    self.logger.critical("Annotation for image: {} doesn't exist, skipping the image".format(image_name))
                    continue

            image = cv2.imread(os.path.join(self.image_dir, image_name))
            cut_points = putils.get_cut_points(image.shape[0], self.out_dim, self.partition)

            for idx_y, y_scan in enumerate(cut_points):
                ymin, _, ymax = y_scan
                for idx_x, x_scan in enumerate(cut_points):
                    xmin, _, xmax = x_scan
                    # If not test mode generate annotations
                    if not self.testmode:
                        anf_out_list = self.get_annot_list(file_name, image, xmin, ymin, xmax, ymax)
                        
                        if len(anf_out_list) == 0:
                            continue
                    
                        annot_file  = "{}_{}_{}{}".format(file_name, idx_y, idx_x, '.box')
                        annot_out_path = os.path.join(self.output_annot_dir, annot_file)
                        anf_out = open(annot_out_path, 'w')
                        for line in anf_out_list:
                            anf_out.write("{}\t{}\t{}\t{}\n".format(line[0], line[1], line[2], line[3]))
                        anf_out.close()

                    image_crop = image[ymin:ymax, xmin:xmax]
                    outfile_name = "{}_{}_{}{}".format(file_name, idx_y, idx_x, ext)
                    cv2.imwrite(os.path.join(self.output_dir, outfile_name), image_crop)


if __name__ == '__main__':
    # Initialize logger
    from mylogger import init_logger
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    init_logger('PROTEIN', log_level)

    parser = argparse.ArgumentParser(description="Protein Object Detection")
    parser.add_argument('-i', '--image_dir', action='store', dest='image_dir', required=True,  help='Direcotry containing images')
    parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', required=True, help='Output direcotry')
    parser.add_argument('-d', '--dim', action='store', dest='out_dim', type=int, default=300, help='Output dimension')
    parser.add_argument('-p', '--partition', action='store', dest='partition', type=int, default=16, help='Image partitions')
    
    parser.add_argument('--testmode', dest='testmode', default=False, action='store_true', help='Test mode or Train/Validation')
    parser.add_argument('-a', '--annotation_dir', action='store', dest='annot_dir', help='Direcotry containing annotation')
    parser.add_argument('-l', '--output_annot_dir', action='store', dest='output_annot_dir', help='Output direcotry for annotation')
    parser.add_argument('-t', '--type', action='store', dest='annot_type', default='box', help='Annotation format')
    parser.add_argument('-f', '--factor', action='store', dest='factor', default=0.1, type=float,  help='BBox tight factor')

    args = parser.parse_args()
    print(args.testmode)
    data_prep = DataPreperation(args.image_dir, args.annot_dir, 
                    args.annot_type, args.output_dir, args.output_annot_dir, args.out_dim, args.partition,
                    args.testmode, args.factor)
    data_prep.process()
    


