import numpy as np
def process_yolo(yolo_outputs, anchors='full', num_classes=80, image_shape=(720,1280), max_boxes=20, score_threshold=0.6, iou_threshold=0.5):
    image_shape = np.array(image_shape)
    if anchors is 'full':
        anchors = np.array([[10.,13.],[16., 30.],[33.,23.],[30.,61.],[62.,45.],[59.,119.],[116.,90.],[156.,198.],[372.,326.]])
    elif anchors is 'tiny':
        anchors = np.array([[10.,14.],[23., 27.],[37.,58.],[81.,82.],[135.,169.],[344,319]])
    
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    def min(x, axis=None, keepdims=False):
        if isinstance(axis, list):
            axis = tuple(axis)
        return np.min(x, axis=axis, keepdims=keepdims)
    def head(feats, anchors, num_classes, input_shape, calc_loss=False):
        """Convert final layer features to bounding box parameters."""
        num_anchors = len(anchors)
        # Reshape to batch, height, width, num_anchors, box_params.
        anchors_tensor = np.reshape(anchors, [1, 1, 1, num_anchors, 2])

        grid_shape = feats.shape[1:3] # height, width   
        grid_y = np.tile(np.reshape(np.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
            [1, grid_shape[1], 1, 1])
        grid_x = np.tile(np.reshape(np.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
            [grid_shape[0], 1, 1, 1])
        
        grid = np.concatenate([grid_x, grid_y],axis=-1)
        grid = grid.astype(feats.dtype)
    
        feats = np.reshape(
            feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
        
        # Adjust preditions to each spatial grid point and anchor size.

        
        box_xy = (sigmoid(feats[..., :2]) + grid) / np.array(grid_shape[::-1]).astype(feats.dtype)
        box_wh = np.exp(feats[..., 2:4]) * anchors_tensor / np.array(input_shape[::-1]).astype(feats.dtype)
        box_confidence = sigmoid(feats[..., 4:5])
        box_class_probs = sigmoid(feats[..., 5:])
    
        if calc_loss == True:
            return grid, feats, box_xy, box_wh
        return box_xy, box_wh, box_confidence, box_class_probs
    def correct_boxes(box_xy, box_wh, input_shape, image_shape):
        '''Get corrected boxes'''
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = input_shape.astype(box_yx.dtype)
        image_shape = image_shape.astype(box_yx.dtype)
        new_shape = np.round(image_shape * min(input_shape/image_shape))
        offset = (input_shape-new_shape)/2./input_shape
        scale = input_shape/new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale
    
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes =  np.concatenate([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ], axis=-1)
        
        # Scale boxes back to original image shape.
        boxes = boxes*np.concatenate([image_shape, image_shape],axis=-1)
        return boxes
    def boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
        '''Process Conv layer output'''
        box_xy, box_wh, box_confidence, box_class_probs = head(feats,
            anchors, num_classes, input_shape)
        boxes = correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = np.reshape(boxes, [-1, 4])
        box_scores = box_confidence * box_class_probs
        box_scores = np.reshape(box_scores, [-1, num_classes])
        return boxes, box_scores
    def boolean_mask(box, mask):
        return box[mask]
    def nms(boxes, scores, maximum_no, overlapThresh):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []
    
        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
    
        # initialize the list of picked indexes	
        pick = []
    
        # grab the coordinates of the bounding boxes
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
    
        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(scores)
    
        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
    
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
    
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
    
            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
    
            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlapThresh)[0])))
    
        # return only the bounding boxes that were picked using the
        # integer data type
        #all_selected = boxes[pick].astype("int")
        if len(pick)<=maximum_no:
            return pick
        else:
            return pick[0:maximum_no]
    
    num_layers = len(yolo_outputs)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
    input_shape = np.array(yolo_outputs[0].shape[1:3]) * 32
    
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = np.concatenate(boxes, axis=0)
    box_scores = np.concatenate(box_scores, axis=0)

    
    
    mask = box_scores >= score_threshold
    max_boxes_tensor = max_boxes
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        class_boxes = boolean_mask(boxes, mask[:, c])
        class_box_scores = boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = nms(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold)
        class_boxes = np.take(class_boxes, nms_index, axis=0)
        class_box_scores = np.take(class_box_scores, nms_index, axis=0)
        classes = c*np.ones(class_box_scores.shape, dtype=np.int32)
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = np.concatenate(boxes_, axis=0)
    scores_ = np.concatenate(scores_, axis=0)
    classes_ = np.concatena te(classes_, axis=0)
    
    return boxes_, scores_, classes_
