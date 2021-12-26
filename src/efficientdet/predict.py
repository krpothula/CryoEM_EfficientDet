import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
import inference
import hparams_config
import utils

class Predict(object):
    def __init__(self, saved_model_dir, input_image, model_name, min_score_thresh, 
                    max_boxes_to_draw, batch_size, output_prediction, hparams, output_image_dir):
        self.saved_model_dir = saved_model_dir
        self.input_image = input_image
        self.model_name = model_name
        self.min_score_thresh = min_score_thresh
        self.max_boxes_to_draw = max_boxes_to_draw
        self.batch_size = batch_size
        self.output_prediction = output_prediction
        self.hparams = hparams
        self.output_image_dir = output_image_dir
        
        self.logdir='/tmp/deff/'
        self.nms_method = 'hard'

        # Create output prediction directory
        os.makedirs(os.path.dirname(self.output_prediction), exist_ok=True)

        # Set image write flag
        if self.output_image_dir is not None:
            self.print_output = True
            os.makedirs(self.output_image_dir, exist_ok=True)
        else:
            self.print_output = False


        self.model_config = hparams_config.get_detection_config(self.model_name)
        if self.hparams is not None:
            self.model_config.override(self.hparams)  # Add custom overrides
        self.model_config.is_training_bn = False
        self.model_config.image_size = utils.parse_image_size(self.model_config.image_size)
        
        self.model_config.nms_configs.score_thresh=self.min_score_thresh
        self.model_config.nms_configs.max_output_size=self.max_boxes_to_draw
        self.model_config.nms_configs.method=self.nms_method

        self.config_dict = {}
        self.config_dict['line_thickness'] = 2
        self.config_dict['max_boxes_to_draw'] = self.max_boxes_to_draw
        self.config_dict['min_score_thresh'] = self.min_score_thresh

        self.driver = inference.ServingDriver(
                    self.model_name,
                    ckpt_path=None,
                    batch_size=self.batch_size,
                    use_xla=False,
                    model_params=self.model_config.as_dict(),
                    **self.config_dict)
        self.driver.load(self.saved_model_dir)

    def process(self):
        all_files = list(tf.io.gfile.glob(self.input_image))
        print("Number of images to process: {}".format(len(all_files)))
        num_batches = (len(all_files) + self.batch_size - 1) // self.batch_size
        pbar = tqdm(range(num_batches))
        for i in pbar:
            batch_files = all_files[i * self.batch_size:(i + 1) * self.batch_size]
            height, width = self.model_config.image_size
            images = [Image.open(f) for f in batch_files]
            # Resize only if images in the same batch have different sizes.
            if len(set([m.size for m in images])) > 1:
                images = [m.resize(height, width) for m in images]
            raw_images = [np.array(m) for m in images]
            size_before_pad = len(raw_images)
            if size_before_pad < self.batch_size:
                padding_size = self.batch_size - size_before_pad
                raw_images += [np.zeros_like(raw_images[0])] * padding_size

            detections_bs = self.driver.serve_images(raw_images)
            
            for j in range(size_before_pad):
                file_name = os.path.basename(batch_files[j])
                prediction = pd.DataFrame(detections_bs[j][detections_bs[j][:, 5]>0.5, 1:], columns=['ymin', 'xmin', 'ymax', 'xmax', 'score', 'class'])
                prediction['FileName'] = file_name
                prediction['class'] = prediction['class'].astype(int)
                prediction = prediction[['FileName', 'class', 'score', 'xmin', 'ymin', 'xmax', 'ymax']]
                with open(self.output_prediction, 'a') as f:
                    prediction.to_csv(f, mode='a', header=f.tell()==0, index=False, line_terminator='\n')

                if self.print_output:
                    img = self.driver.visualize(raw_images[j], detections_bs[j], **self.config_dict)
                    output_image_path = os.path.join(self.output_image_dir, file_name)
                    Image.fromarray(img).save(output_image_path)
                    # print('writing file to %s' % output_image_path)
                    pbar.set_description(output_image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Protein Object Detection")
    parser.add_argument('--saved_model_dir', action='store', dest='saved_model_dir', required=True, help='')
    parser.add_argument('--input_image', action='store', dest='input_image', required=True, help='')
    parser.add_argument('--hparams', action='store', dest='hparams', default=None,  help='')
    parser.add_argument('--output_image_dir', action='store', dest='output_image_dir', default=None,  help='')
    parser.add_argument('--model_name', action='store', dest='model_name', default='efficientdet-d0',  help='')
    parser.add_argument('--min_score_thresh', action='store', dest='min_score_thresh', type=float, default=0.5,  help='')
    parser.add_argument('--max_boxes_to_draw', action='store', dest='max_boxes_to_draw', type=int, default=100,  help='')
    parser.add_argument('--batch_size', action='store', dest='batch_size', type=int, default=1,  help='')
    parser.add_argument('--line_thickness', action='store', dest='line_thickness', type=int, default=2,  help='')
    parser.add_argument('--output_prediction', action='store', dest='output_prediction', default='predictions.csv', help='')

    args = parser.parse_args()
    predict = Predict(args.saved_model_dir, args.input_image, args.model_name, args.min_score_thresh, 
                    args.max_boxes_to_draw, args.batch_size, args.output_prediction, args.hparams, args.output_image_dir)
    predict.process()
    

