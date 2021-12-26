import os

def get_filename_ext(file_name_ext):
    file_name, ext = os.path.splitext(file_name_ext)
    return file_name, ext

def get_cut_points(dim, out_dim, partition):
    surplus = out_dim * partition - dim
    overlap = surplus / (partition - 1)
    cut_points = [[0, out_dim, out_dim]]
    for _ in range(partition - 1):
        start = cut_points[-1][1] - overlap
        end = start + out_dim
        cut_points.append([round(start), end, round(end)])

    return cut_points

def get_box_coordinates(line, img_hight, factor=0.1):
    x = int(float(line[0]))
    # Invert the Y coordinate suitabel for computer display
    y = img_hight - int(float(line[1]))
    width = max(int(line[2]), 0)
    height = max(int(line[3]), 0)

    # Tight the box based on the given factor
    x = int(x + width * factor)
    y = int(y - height * factor)
    width = int(width * (1 - 2 * factor))
    height = int(height * (1 - 2 * factor))

    xmin = x
    ymin = y - height
    xmax = xmin + width
    ymax = y
    return xmin, ymin, xmax, ymax

def get_overlap(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    overlap = interArea / boxBArea
    return overlap

