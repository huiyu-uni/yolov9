import random
import cv2
import os
from pathlib import Path
import numpy as np
import torch
import time

def read_random_jpg_from_folder(folder_path):
    """
    Randomly selects and reads a JPG image from the specified folder.

    Args:
        folder_path (str): The path to the folder containing the images.

    Returns:
        numpy.ndarray: The loaded image as a NumPy array (BGR format),
                       or None if no JPG images are found or an error occurs.
        str: The path of the randomly selected image, or None.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return None, None

    # List all files in the directory
    all_files = os.listdir(folder_path)

    # Filter for JPG images (case-insensitive)
    jpg_files = [
        f for f in all_files
        if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')
    ]

    if not jpg_files:
        print(f"No JPG/JPEG images found in folder '{folder_path}'.")
        return None, None

    # Randomly select one JPG file
    random_jpg_filename = random.choice(jpg_files)
    random_jpg_path = os.path.join(folder_path, random_jpg_filename)

    print(f"Selected image: {random_jpg_path}")

    # Read the image using OpenCV
    try:
        image = cv2.imread(random_jpg_path)
        if image is None:
            print(f"Error: Could not read image at '{random_jpg_path}'. Check file corruption or path.")
            return None, None
        return image, random_jpg_path
    except Exception as e:
        print(f"An error occurred while reading the image: {e}")
        return None, None

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y

def find_txt_file(root_path, target_filename):
    
    # tic = time.perf_counter()

    for dirpath, dirnames, filenames in os.walk(root_path):
        if target_filename in filenames:
            # tac = time.perf_counter()
    
            # print(f"Process finding txt file {tac - tic:0.4f} seconds")
            return os.path.join(dirpath, target_filename)

    # tac = time.perf_counter()
    
    # print(f"Process finding txt file {tac - tic:0.4f} seconds")
    
    return None

def unlabeled_dataset_stats(img_paths, pseudo_labels_dir):
    # calculate stats 
    with open(img_paths, 'r') as file:
        lines = file.readlines()

    moth_counter_dict = {
        str(0.1) : int(0),
        str(0.2) : int(0),
        str(0.3) : int(0),
        str(0.4) : int(0),
        str(0.5) : int(0),
        str(0.6) : int(0),
        str(0.7) : int(0),
        str(0.8) : int(0),
        str(0.9) : int(0),
    }
    
    non_moth_counter_dict = {
        str(0.1) : int(0),
        str(0.2) : int(0),
        str(0.3) : int(0),
        str(0.4) : int(0),
        str(0.5) : int(0),
        str(0.6) : int(0),
        str(0.7) : int(0),
        str(0.8) : int(0),
        str(0.9) : int(0),
    }
    
    confidences = [0.1, 0.2, 0.3, 0.4, 0.5 ,0.6, 0.7, 0.8, 0.9]

    from tqdm import tqdm
    
    for iter, img_path in enumerate(tqdm(lines)):
        img_path = img_path.strip()
        img_full_path, ext = img_path.rsplit('.', 1)
        _, img_name = img_full_path.rsplit('/', 1)
        
        # Find pseudo labels path
        labels_path = img_name + '.txt'
        
        # print(img_path)
        # print(img_full_path)
        # print(labels_path)
        # print(pseudo_labels_dir)
        
        lb_file = find_txt_file(pseudo_labels_dir, labels_path)
        if lb_file:
            with open(lb_file) as f:
                lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
        
                for conf in confidences:
                    for each_label in lb:   
                        category, score = each_label[0], each_label[5]
                        
                        if category == 0 and score > conf:
                            moth_counter_dict[str(conf)] += 1
                        if category == 1 and score > conf:
                            non_moth_counter_dict[str(conf)] += 1
          
        if iter % 300 == 0:
            print(moth_counter_dict)
            print(non_moth_counter_dict)
                    
    print(moth_counter_dict)
    print(non_moth_counter_dict)
        
    ### conf=0.3 and iou=0.5
    # {'0.1': 228984, '0.2': 228984, '0.3': 228984, '0.4': 222804, '0.5': 217672, '0.6': 211865, '0.7': 203309, '0.8': 176412, '0.9': 28613}
    # {'0.1': 73970 , '0.2': 73970 , '0.3': 73970 , '0.4': 61318 , '0.5': 50297 , '0.6': 38797 , '0.7': 25637 , '0.8': 10062 , '0.9': 246  }



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def filter_and_recalculate_pseudo_labels(pseudo_labels):
    """
    Filters pseudo-labels for category 0 with score > 0.7 and recalculates
    their coordinates relative to the largest bounding box containing these instances.

    Args:
        pseudo_labels (list): A list of dictionaries, where each dictionary
                              represents a pseudo-label with keys:
                              'category' (int), 'xyxy' (list/tuple of 4 floats [x1, y1, x2, y2]),
                              and 'score' (float).

    Returns:
        tuple: A tuple containing:
            - list: A new list of dictionaries with the original pseudo-labels
                    but with 'xyxy' coordinates updated to be relative to the
                    newly calculated largest region. If no instances meet the
                    criteria, an empty list is returned.
            - list: The absolute coordinates [x1, y1, x2, y2] of the largest
                    bounding box that contains all filtered instances.
                    Returns [0.0, 0.0, 0.0, 0.0] if no instances meet criteria.
    """
    filtered_instances_coords = []
    
    # Step 1: Identify instances of category 0 with score higher than 0.7
    for label in pseudo_labels:
        category = label.get('category')
        score = label.get('score')
        xyxy = label.get('xyxy')

        if category is None or score is None or xyxy is None or not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
            print(f"Warning: Skipping malformed pseudo-label: {label}")
            continue

        if category == 0 and score > 0.7:
            filtered_instances_coords.append(xyxy)

    if not filtered_instances_coords:
        print("No instances of category 0 with score > 0.7 found.")
        return [], [0.0, 0.0, 0.0, 0.0]

    import math
    # Step 2: Find the largest region (bounding box) containing these instances
    min_x1 = math.inf
    min_y1 = math.inf
    max_x2 = -math.inf
    max_y2 = -math.inf

    for x1, y1, x2, y2 in filtered_instances_coords:
        min_x1 = min(min_x1, x1)
        min_y1 = min(min_y1, y1)
        max_x2 = max(max_x2, x2)
        max_y2 = max(max_y2, y2)

    # The new rectangle (largest region)
    largest_region_bbox = [min_x1, min_y1, max_x2, max_y2]
    
    # Calculate width and height of the largest region
    region_width = max_x2 - min_x1
    region_height = max_y2 - min_y1

    if region_width == 0 or region_height == 0:
        print("Warning: The largest region has zero width or height. Relative coordinates will be undefined or zero.")
        # Handle cases where width/height is zero to avoid division by zero
        # If width/height is 0, all points are on a line or a single point.
        # We can return [0,0,0,0] for relative coordinates in such cases.
        recalculated_pseudo_labels = []
        for label in pseudo_labels:
            new_label = label.copy()
            new_label['xyxy'] = [0.0, 0.0, 0.0, 0.0] # Set to zero if region is degenerate
            recalculated_pseudo_labels.append(new_label)
        return recalculated_pseudo_labels, largest_region_bbox


    # Step 3: Recalculate coordinates relative to the new rectangle for ALL original pseudo-labels
    recalculated_pseudo_labels = []
    for label in pseudo_labels:
        original_xyxy = label.get('xyxy')
        
        # Create a copy to avoid modifying the original input list directly
        new_label = label.copy() 
        
        if original_xyxy is None or not isinstance(original_xyxy, (list, tuple)) or len(original_xyxy) != 4:
            print(f"Warning: Skipping recalculation for malformed xyxy in label: {label}")
            new_label['xyxy'] = [0.0, 0.0, 0.0, 0.0] # Default to zero if malformed
        else:
            x1_orig, y1_orig, x2_orig, y2_orig = original_xyxy

            # Calculate new relative coordinates
            new_x1 = (x1_orig - min_x1) / region_width
            new_y1 = (y1_orig - min_y1) / region_height
            new_x2 = (x2_orig - min_x1) / region_width
            new_y2 = (y2_orig - min_y1) / region_height

            # Ensure coordinates are within [0, 1] range, clamping if necessary due to floating point inaccuracies
            new_label['xyxy'] = [
                max(0.0, min(1.0, new_x1)),
                max(0.0, min(1.0, new_y1)),
                max(0.0, min(1.0, new_x2)),
                max(0.0, min(1.0, new_y2))
            ]
        recalculated_pseudo_labels.append(new_label)

    return recalculated_pseudo_labels, largest_region_bbox

def load_crops_from_unlabeled_dataset(img_paths, pseudo_labels_dir, num_imgs=5, pad=100, zoom_factor=0.5, threshold_list=[0.7, 1.0]):
    
    # moth / non-moths crops
    crops = []
    
    with open(img_paths, 'r') as file:
        lines = file.readlines()


    # Choose 5 random lines without replacement
    random_img_paths = random.sample(lines, num_imgs)

    tic = time.perf_counter()

    # Print the random lines
    for img_path in random_img_paths:
        img_path = img_path.strip()
        img_full_path, ext = img_path.rsplit('.', 1)
        _, img_name = img_full_path.rsplit('/', 1)
        
        # Find pseudo labels path
        labels_path = img_name + '.txt'
        
        
        lb_file = find_txt_file(pseudo_labels_dir, labels_path)
        
        if lb_file:
            with open(lb_file) as f:
                lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

            tec = time.perf_counter()

            im = cv2.imread(img_path)  # BGR
            BGR = True
            toc = time.perf_counter()
            
            # Load labels with formatting [label, u1, v1, w, h, score]
            # 0: Moth, 1: Non-Moth
            pseudo_labels = []
            print(lb)
            for each_label in lb:
                category, xyxy, score = each_label[0], xywhn2xyxy(each_label[1:5], im.shape[1], im.shape[0]), each_label[5]
                pseudo_labels.append({
                    'category': category, 'xyxy': xyxy.tolist(), 'score': score
                })
            print(pseudo_labels)
            
            
            # recalculated_labels, new_bounding_box = filter_and_recalculate_pseudo_labels(pseudo_labels)
            
            # print(recalculated_labels)
            # print(new_bounding_box)
            
            
            for each_idx, each_label in enumerate(lb):
                crops_list = [pl['xyxy'] for pl in pseudo_labels]
                
                category, xyxy, score = each_label[0], xywhn2xyxy(each_label[1:5], im.shape[1], im.shape[0]), each_label[5]

                # Segment an area where there is maximum number of high confidence insects

                ### TODO: Select only moth crops with 0.7 confidence
                if (category == 0 and score > threshold_list[0]) or (category == 1 and score > threshold_list[1]):
                # if (category == 1 and score > 0.8):
                # if (category == 0 and score > 0.7):
                    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
                    xyxy = torch.tensor(xyxy).view(-1, 4)
                    clip_boxes(xyxy, im.shape)
                    
                    ### TODO: Add paddings to rectangular crops to mitigate edge effect 
                    min_x = int(xyxy[0, 1]-pad)
                    max_x = int(xyxy[0, 3]+pad)
                    min_y = int(xyxy[0, 0]-pad)
                    max_y = int(xyxy[0, 2]+pad)
                    
                    # Check whether the padded crop is still within the original image
                    if min_x > 0 and min_y > 0 and max_x < im.shape[0] and max_y < im.shape[1]:
                        crop = im[min_x:max_x, min_y:max_y, ::(1 if BGR else -1)]
                    else: continue
                    
                    
                    # Check whether there are other instances within the crop
                    del crops_list[each_idx]
                    # print(f'xyxy: {xyxy}')
                    # print(f'crops_list: {crops_list}')
                    # print(f'box_iou: {box_iou(xyxy, torch.tensor(crops_list).view(-1, 4))}')
                    if len(crops_list) > 1 and torch.max(box_iou(xyxy, torch.tensor(crops_list).view(-1, 4))) > 0.0:
                        continue
                    
                    ### Resize
                    crop = cv2.resize(crop, (0,0), fx=zoom_factor, fy=zoom_factor)      
                    crops.append([crop, category])
                    # bboxes.append([[int(xyxy[0, 0]), int(xyxy[0, 1]), int(xyxy[0, 2]), int(xyxy[0, 3])], category])
            # if new_bounding_box:
            #     new_bounding_box = torch.tensor(new_bounding_box).view(-1, 4)
            #     clip_boxes(new_bounding_box, im.shape)
            #     crop = im[int(new_bounding_box[0, 1]):int(new_bounding_box[0, 3]), int(new_bounding_box[0, 0]):int(new_bounding_box[0, 2]), ::(1 if BGR else -1)]
            #     cv2.imwrite('patch.png', crop)
            print(f"Process image loading in {toc - tec:0.4f} seconds")
        
    return crops
    
    
def paste_in(img, crops, max_crops=None, orig_labels=[], max_iou=0.0, pad=100, zoom_factor=0.5):
    output_img_orig = img.copy()
    output_img_normal = img.copy()
    output_img_mixed = img.copy()

    labels = []
    h_base, w_base = output_img_orig.shape[:2]

    if max_crops:
        crops = random.sample(crops, min(max_crops, len(crops)))
        
    if orig_labels:
        orig_labels = torch.tensor(np.array(orig_labels)[:, 1:]).view(-1, 4)
        orig_labels = xywhn2xyxy(orig_labels, w_base, h_base)
    
    picked_boxes = []

    attempts = 0
    num_iterations = 0

    for each_crop in crops:
        crop, category = each_crop
        h_crop, w_crop = crop.shape[:2]
        
        # Ensure crop fits within the base image
        if h_crop > h_base or w_crop > w_base:
            attempts += 1
            continue  # skip crop if too big
        
        # Random position within bounds
        x = random.randint(0, w_base - w_crop)
        y = random.randint(0, h_base - h_crop)
        
        xyxy_lst = [float(x), float(y), float(x + w_crop), float(y + h_crop)]
        xyxy = torch.tensor(xyxy_lst).view(-1, 4)
        
        # if type(orig_labels) != list:
        
        if orig_labels.size and torch.max(box_iou(xyxy, orig_labels)) > max_iou:
            attempts += 1
            continue
        
        if picked_boxes and torch.max(box_iou(xyxy, torch.tensor(picked_boxes).view(-1, 4))) > max_iou:
            attempts += 1
            continue
            
        picked_boxes.append(xyxy_lst)
        # # Paste the crop
        output_img_orig[y:y+h_crop, x:x+w_crop] = crop
        
        
        mask = 255 * np.ones(crop.shape[:2], dtype=np.uint8)
        center = (x + int(w_crop / 2), y + int(h_crop / 2))
        output_img_normal = cv2.seamlessClone(crop, output_img_normal, mask, center, cv2.NORMAL_CLONE)
        output_img_mixed = cv2.seamlessClone(crop, output_img_mixed, mask, center, cv2.MIXED_CLONE)
        
        
        ## orig-crop --> output_img_orig[y:y+h_crop, x:x+w_crop] 
        ## hard-crop --> crop 
        ## smooth-crop --> output_img_normal[y:y+h_crop, x:x+w_crop] 
        
        num_iterations += 1
        
        print(xyxy)
        xyxy[0][0] += pad*zoom_factor
        xyxy[0][1] += pad*zoom_factor
        xyxy[0][2] -= pad*zoom_factor
        xyxy[0][3] -= pad*zoom_factor
        
        lst = [float(category)]
        lst.extend(xyxy2xywhn(xyxy, w_base, h_base)[0].tolist())
        labels.append(lst)

    print(f'The number of attempts are: {attempts}')

    return output_img_orig, output_img_normal, output_img_mixed, labels

def draw_bboxes_uv(image_path, bbox_list, output_path=None):
    """
    Draw bounding boxes (UV normalized) on an image.

    Parameters:
        image_path (str): Path to input image.
        bbox_list (list): List of [category, x_norm, y_norm, w_norm, h_norm]
        output_path (str, optional): Path to save the image.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    height, width = image.shape[:2]

    for bbox in bbox_list:
        cate, x_norm, y_norm, w_norm, h_norm = bbox

        # Convert to absolute pixel values
        x = int(x_norm * width)
        y = int(y_norm * height)
        w = int(w_norm * width)
        h = int(h_norm * height)

        top_left = (x - int(w/2), y - int(h/2))
        bottom_right = (x + int(w/2), y + int(h/2))

        # Draw box
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        # Draw label
        cv2.putText(image, str(cate), (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save or display image
    if output_path:
        cv2.imwrite(output_path, image)
    else:
        cv2.imshow("BBoxes", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def caching_balance(crops, new_crops, max_size=100):
    # Caching crops with fixed size of 100
    crops.extend(new_crops)
    
    if len(crops) > max_size:
        diff = len(crops) - max_size
        del crops[:diff]
    return crops


num_imgs = 100
crops = []
for i in range(num_imgs):  # runs 10 times
    print(f"Loop iteration {i + 1}")
    
    # img = cv2.imread("/home/yu/Documents/AMMOD/ami-trap-dataset/Archive/ami_traps/ami_traps_dataset/images/01-20220930005959-snapshot.jpg")  # BGR
    # with open("/home/yu/Documents/AMMOD/ami-trap-dataset/Archive/ami_traps/ami_traps_dataset/images/01-20220930005959-snapshot.txt") as f:
    img = cv2.imread("/home/yu/Documents/AMMOD/ami-trap-dataset/Archive/ami_traps/ami_traps_dataset/images/20220903213444-00-10.jpg")
    with open("/home/yu/Documents/AMMOD/ami-trap-dataset/Archive/ami_traps/ami_traps_dataset/images/20220903213444-00-10.txt") as f:
    # img = cv2.imread("/home/yu/Documents/AMMOD/nid-dataset/nid-yolo/2021-06-22_Weinschale_2819.JPG")
    # with open("/home/yu/Documents/AMMOD/nid-dataset/nid-yolo/2021-06-22_Weinschale_2819.txt") as f:
    
    # img, image_path = read_random_jpg_from_folder("/home/yu/Documents/AMMOD/ami-trap-dataset/Archive/ami_traps/ami_traps_dataset/images")
    # img, image_path = read_random_jpg_from_folder("/home/yu/Documents/AMMOD/nid-dataset/nid-yolo")
    # with open(image_path.replace("jpg","txt")) as f:
    
        orig_labels = [x.split() for x in f.read().strip().splitlines()]  # labels
        for index, each_label in enumerate(orig_labels):
            orig_labels[index] = [float(x) for x in each_label]

    tic = time.perf_counter()


    new_crops = load_crops_from_unlabeled_dataset(img_paths='/home/yu/Documents/AMMOD/nid-dataset/unlabeled.txt',
                                        pseudo_labels_dir='/home/yu/Documents/AMMOD/yolov9/runs/detect/pseudo-labels',
                                        num_imgs=1,
                                        pad=100,
                                        zoom_factor=0.5,
                                        threshold_list=[0.7, 1.0])

    ### TODO: Create cache for crops, if the crops exceeds maximum number. Then remove the first n crops
    print(f'len_new_crops: {len(new_crops)}')
    crops = caching_balance(crops, new_crops, max_size=100)

    print(f'cache_crops: {len(crops)}')

    tac = time.perf_counter()

    ### TODO: Paste-In insect crops to img with certain probability (0.3)
    # print(orig_labels)
    # output_img, output_labels = paste_in(img, crops, 10, orig_labels) 
    output_img_orig, output_img_normal, output_img_mixed, labels = paste_in(img=img, 
                                                                            crops=crops, 
                                                                            max_crops=50, 
                                                                            orig_labels=orig_labels,
                                                                            pad=100,
                                                                            zoom_factor=0.5) 

    print(labels)

    os.makedirs('result', exist_ok=True)

    cv2.imwrite("result/img_orig.jpg", img)
    cv2.imwrite("result/output_img_orig.jpg", output_img_orig)
    cv2.imwrite("result/output_img_normal.jpg", output_img_normal)
    cv2.imwrite("result/output_img_mixed.jpg", output_img_mixed)


    draw_bboxes_uv("result/output_img_orig.jpg", labels, "result/output_img_orig_bboxes.jpg")

    time.sleep(1)
