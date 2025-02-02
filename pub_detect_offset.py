#use conda::base
from ultralytics import YOLO
from PIL import Image
import cv2
import time
import numpy as np
import torch
import time
from collections import deque
import rclpy
from std_msgs.msg import Float32MultiArray
from scipy.optimize import least_squares
from find_cam import find_cam
last_mos = np.array([0,0,0])
last_mos_queue = deque(maxlen=10)
TOOL_CENTER = np.array([180,291])
def process_results2(results):
    transform = np.array([0.01, 0.01, 1])
    global last_mos
    if len(results) == 1:
        result = results[0]
        if not (results[0].keypoints.conf is None):
            mid_of_studs = results[0].keypoints.xy[0,1,:]
            mid_of_studs_conf = results[0].keypoints.conf[0,1]
            last_mos = 0.8 * last_mos + 0.2 * np.array(mid_of_studs.to('cpu'))
            output = np.zeros(3)
            output[:2] = last_mos
            output[2] = mid_of_studs_conf.cpu().item()
            #output = output @ transform.T
            print("center: ", last_mos, "with confidence", mid_of_studs_conf.cpu().item())
        else:
            print("no lego keypoints detected")
            output = np.zeros(3)
        
        return output

def min_enclosing_circle_tangent_to_lines(contour, line_x, line_y):
    """
    Find the minimum enclosing circle of a contour that is tangent to both a vertical and a horizontal line.
    
    Parameters:
        contour (np.ndarray): Contour points as a numpy array of shape (n, 2).
        line_x (float): X-coordinate of the vertical line.
        line_y (float): Y-coordinate of the horizontal line.
    
    Returns:
        tuple: (center, radius) where center is (x, y) and radius is the circle radius.
    """
    # Find the initial minimum enclosing circle
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    # Adjust the x-coordinate of the circle center for vertical line tangency
    if cx < line_x:
        cx = line_x -radius
    else:
        cx = line_x + radius

    # Adjust the y-coordinate of the circle center for horizontal line tangency
    if cy < line_y:
        cy = line_y -radius
    else:
        cy = line_y + radius

    # Recalculate the radius to ensure all points are enclosed
    distances = np.sqrt((contour[:, 0] - cx) ** 2 + (contour[:, 1] - cy) ** 2)
    radius = np.max(distances)
    if cx < line_x:
        cx = line_x -radius
    else:
        cx = line_x + radius
    if cy < line_y:
        cy = line_y -radius
    else:
        cy = line_y + radius

    # Return the adjusted circle center and radius
    return np.array((cx, cy)).astype(int), int(radius)

def find_intersections(centers, y_top, y_bottom):
    """
    Calculate the intersection points of Line1 with horizontal lines at y_top and y_bottom.
    
    Parameters:
        centers (list of tuples): Two points [(x0, y0), (x1, y1)] defining Line1.
        y_top (int): Y-coordinate of the horizontal line (Line2).
        y_bottom (int): Y-coordinate of the horizontal line (Line3).
    
    Returns:
        tuple: Intersection points ((x_intersect1, y_top), (x_intersect2, y_bottom)).
    """
    # Extract points for Line1
    (x0, y0), (x1, y1) = centers

    # Calculate slope (m1) and intercept (c1) of Line1
    if x1 != x0:  # Line1 is not vertical
        m1 = (y1 - y0) / (x1 - x0)
        c1 = y0 - m1 * x0
    else:  # Line1 is vertical
        m1 = float('inf')  # Infinite slope
        c1 = x0  # Vertical line's x-coordinate

    # Intersection with Line2 (y = y_top)
    if m1 != float('inf'):
        x_intersect1 = (y_top - c1) / m1
    else:
        x_intersect1 = c1  # For vertical Line1
    intersection1 = (int(round(x_intersect1)), y_top)

    # Intersection with Line3 (y = y_bottom)
    if m1 != float('inf'):
        x_intersect2 = (y_bottom - c1) / m1
    else:
        x_intersect2 = c1  # For vertical Line1
    intersection2 = (int(round(x_intersect2)), y_bottom)

    return np.asarray([intersection1, intersection2])

def process_results(result, conf, angle):
    transform = np.array([0.01, 0.01, 1])
    global last_mos_queue
    
    # Convert the new measurement to CPU and append to the queue
    if conf > 0.5:
        new_measurement = np.concatenate([result, [angle]])
        last_mos_queue.append(new_measurement)
        
        # Compute the rolling average
        rolling_average = np.mean(np.array(last_mos_queue), axis=0)

        output = np.zeros(3)
        output[:3] = rolling_average

        print("center: ", rolling_average, "with confidence", conf)
            
        return output
    else:
        return None
    

def compute_offset(camera, model, fx = 1100 , fy = 1100, z = 30.0):
    '''
    Arguments: 
    camera to read from, and yolo keypoint model to use
    camera focal and depth in mm. Assume cx, cy are at image center, and assume fixed z
    Outputs:
    [x,y,confidence] if detected, None otherwise
    '''
    t0 = time.perf_counter()
    ret, og_frame = camera.read()
    # og_frame = cv2.imread("sample.jpg")
    if ret:
        h, w, c = og_frame.shape
    else:
        print("read frame file")
        return None
    start_w = (w - 480) // 2
    end_w = start_w + 480
    
    # Crop the middle section
    # og_frame = cv2.resize(og_frame, [640,480])
    og_frame = og_frame[:, start_w:end_w, :]
    t1 = time.perf_counter()
    results = model.predict(og_frame, show = False, verbose = False)
    
    t2 = time.perf_counter()
    if results[0].masks is None:
        print("No Studs detected")
        return None
    mask = results[0].masks.xy
    conf = torch.min(results[0].boxes.conf).to('cpu').item()
    if (len(results[0].masks) != 2):
        print(len(results[0].masks), " studs detected, abort computing offset")
        return None
    segments = np.zeros(og_frame.shape[:2])
    centers = []
    radiuses = []
    for i in range(min(2, len(mask))):
        (x, y), radius = cv2.minEnclosingCircle(mask[i])
        center = (int(x), int(y))  # Convert center coordinates to integers
        radius = int(radius)       # Convert radius to an integer
        centers.append(center)
        radiuses.append(radius)
    top_stud_mask = mask[np.argmax([np.median(mask[0][:,1]), np.median(mask[1][:,1])])] #top stud is the one with higher Y value, which is at the BOTTOM in image
    bottom_stud_mask = mask[np.argmin([np.median(mask[0][:,1]), np.median(mask[1][:,1])])]
    y_top = np.min(top_stud_mask[:,1]).astype(int)
    y_bottom = np.max(bottom_stud_mask[:,1]).astype(int)
    if(np.average([centers[0][0], centers[1][0]])< 200):
        top_stud_side_x = np.max(top_stud_mask[:,0]).astype(int)
        bottom_stud_side_x = np.max(bottom_stud_mask[:,0]).astype(int)
    else:
        top_stud_side_x = np.min(top_stud_mask[:,0]).astype(int)
        bottom_stud_side_x = np.min(bottom_stud_mask[:,0]).astype(int)

    # intersects = find_intersections(centers, y_top, y_bottom)
    # target_center = np.average(intersects, axis=0)

    top_stud_center_adj, top_stud_radius_adj = min_enclosing_circle_tangent_to_lines(top_stud_mask, top_stud_side_x, y_top)
    bottom_stud_center_adj, bottom_stud_radius_adj = min_enclosing_circle_tangent_to_lines(bottom_stud_mask, bottom_stud_side_x, y_bottom)
    target_center_adj = np.mean([top_stud_center_adj, bottom_stud_center_adj], axis = 0)

    t3 = time.perf_counter()

    studs_diff = top_stud_center_adj - bottom_stud_center_adj
    angle = np.arctan2(studs_diff[1], studs_diff[0])
    angle -= 1.5143
    output = process_results(target_center_adj, conf,angle)
    if output is None:
        print("low confidence, ignoring offset")
        return None
    output[:2] -= TOOL_CENTER#diff from center
    output[:2] *= z
    output[:2] /= np.array([fx, fy])
    output[0] *= -1 #x is reversed
    output[1] *= -1 
    return np.concatenate([output[:2] / 1000, output[2]]) #mm to meter
    # except Exception as e:
    #     print("error: ", e)
    #     return None

model = YOLO("studs-seg2.pt")
camera_ids = find_cam()
if camera_ids is None:
    print("no camera detected")
    assert False
else:
    camera = cv2.VideoCapture(camera_ids[-1])

filtered_center = np.array([0,0])

rclpy.init()
node = rclpy.create_node('tool_offset_publisher')
tool_offset_pub = node.create_publisher(Float32MultiArray, 'tool_offset', 10)

# Set the loop rate (e.g., 10 Hz)
rate = node.create_rate(20)

while node.ok():
    # Get the detected offset
    offset = compute_offset(camera, model)
    print(offset)
    # Publish only if offset is not None
    # If offset is None (no feature detected), controller's offset isn't updated
    if offset is not None:
        # Create a Float32MultiArray message
        msg = Float32MultiArray()
        msg.data = offset
        print("published")
        # Publish the message
        tool_offset_pub.publish(msg)

    # Sleep to maintain the loop rate
    rate.sleep() 
# while True:
    
#     ret, og_frame = camera.read()
#     h, w, c = og_frame.shape

# # Calculate the start and end points for the width dimension to get the middle 480 pixels
#     start_w = (w - 480) // 2
#     end_w = start_w + 480

#     # Crop the middle section
#     og_frame = og_frame[:, start_w:end_w, :]
#     results = model.predict(og_frame, show = True, verbose = False)
#     process_results(results)