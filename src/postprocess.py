from genericpath import exists
import os
import argparse
import logging
import putils
import pandas as pd
import tensorflow as tf

# Initialize logger
from mylogger import init_logger
log_level = os.getenv('LOG_LEVEL', 'INFO')
init_logger('PROTEIN', log_level)


class Postprocess(object):
    def __init__(self, output_prediction, final_prediction, final_annotation_dir, 
                    annot_type, out_dim, partition, image_dim):
        self.logger = logging.getLogger('PROTEIN')
        self.output_prediction = output_prediction
        self.final_prediction = final_prediction
        self.final_annotation_dir = final_annotation_dir
        self.annot_type = annot_type
        self.out_dim = out_dim
        self.partition = partition
        self.image_dim = image_dim
        os.makedirs(self.final_annotation_dir, exist_ok=True)

    def non_max_suppression(self, data):
        selected_idx = tf.image.non_max_suppression(
            data[['xmin_new', 'ymin_new', 'xmax_new', 'ymax_new']], 
            data['score'], max_output_size=len(data), iou_threshold=0.5)
        return data.iloc[selected_idx]
        
    def process(self):
        output_pred = pd.read_csv(self.output_prediction)

        def get_root_filename(file_name):
            file_name, ext = os.path.splitext(file_name)
            file_name, party, partx = file_name.rsplit('_', 2)
            file_name = file_name + ext
            return file_name, int(party), int(partx)

        output_pred['FileName_unpack'] = output_pred['FileName'].apply(lambda x: get_root_filename(x))
        output_pred['root_file'] = output_pred['FileName_unpack'].apply(lambda x: x[0])
        output_pred['party'] = output_pred['FileName_unpack'].apply(lambda x: x[1])
        output_pred['partx'] = output_pred['FileName_unpack'].apply(lambda x: x[2])

        cut_points = putils.get_cut_points(self.image_dim, self.out_dim, self.partition)
        output_pred['xoffset'] = output_pred['partx'].apply(lambda x: cut_points[x][0])
        output_pred['yoffset'] = output_pred['party'].apply(lambda x: cut_points[x][0])
        output_pred['xmin_new'] = output_pred['xmin'] + output_pred['xoffset']
        output_pred['xmax_new'] = output_pred['xmax'] + output_pred['xoffset']
        output_pred['ymin_new'] = output_pred['ymin'] + output_pred['yoffset']
        output_pred['ymax_new'] = output_pred['ymax'] + output_pred['yoffset']
        output_pred['width'] = output_pred['xmax_new'] - output_pred['xmin_new']
        output_pred['height'] = output_pred['ymax_new'] - output_pred['ymin_new']
        output_pred['ymax_new_flip'] = self.image_dim - output_pred['ymax_new']
        output_pred['ymax_new_flip'] = output_pred['ymax_new_flip'].astype(int)
        output_pred['xmin_new'] = output_pred['xmin_new'].astype(int)
        output_pred['width'] = output_pred['width'].astype(int)
        output_pred['height'] = output_pred['height'].astype(int)
        
        final_output = []
        for grp_id, group in output_pred.groupby(['root_file']):
            group = self.non_max_suppression(group)
            final_output.append(group)
            file_name, ext = putils.get_filename_ext(grp_id)
            file_path = os.path.join(self.final_annotation_dir, file_name + '.' + self.annot_type)
            group[['xmin_new', 'ymax_new_flip', 'width', 'height']].\
                to_csv(file_path, index=False, sep='\t', header=False)
        
        final_output = pd.concat(final_output)
        final_output.to_csv(self.final_prediction, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Protein Object Detection")
    parser.add_argument('--output_prediction', action='store', dest='output_prediction', required=True, help='')
    parser.add_argument('-d', '--dim', action='store', dest='out_dim', type=int, default=300, help='Output dimension')
    parser.add_argument('-p', '--partition', action='store', dest='partition', type=int, default=16, help='Image partitions')
    parser.add_argument('-t', '--type', action='store', dest='annot_type', default='box', help='Annotation format')
    parser.add_argument('--image_dim', action='store', dest='image_dim', type=int, default=4096, help='Image dimension')
    parser.add_argument('--final_prediction', action='store', dest='final_prediction', default='predictions_final.csv', help='')
    parser.add_argument('--final_annotation_dir', action='store', dest='final_annotation_dir', help='')

    
    args = parser.parse_args()
    post_process = Postprocess(args.output_prediction, args.final_prediction, args.final_annotation_dir,
                    args.annot_type, args.out_dim, args.partition, args.image_dim)
    post_process.process()
    


