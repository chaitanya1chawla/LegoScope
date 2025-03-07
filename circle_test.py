#use conda::base
from ultralytics import YOLO
from PIL import Image
import cv2
import time
import numpy as np
import torch
from collections import deque
from scipy.optimize import least_squares
from circle_fit import taubinSVD
from scipy.optimize import minimize

last_mos = np.array([0,0,0])
last_mos_queue = deque(maxlen=10)
TOOL_CENTER = np.array([180,291])

##### Description: #####
# Detects studs in the image
# Returns the offset of the tool center from the average of the two studs' centers
# Returns the angle of the tool (by computing the angle between the two studs)


# def process_results2(results):
#     transform = np.array([0.01, 0.01, 1])
#     global last_mos
#     if len(results) == 1:
#         result = results[0]
#         if not (results[0].keypoints.conf is None):
#             mid_of_studs = results[0].keypoints.xy[0,1,:]
#             mid_of_studs_conf = results[0].keypoints.conf[0,1]
#             last_mos = 0.8 * last_mos + 0.2 * np.array(mid_of_studs.to('cpu'))
#             output = np.zeros(3)
#             output[:2] = last_mos
#             output[2] = mid_of_studs_conf.cpu().item()
#             #output = output @ transform.T
#             print("center: ", last_mos, "with confidence", mid_of_studs_conf.cpu().item())
#         else:
#             print("no lego keypoints detected")
#             output = np.zeros(3)
        
#         return output

# def fit_circle2(points):
#     """
#     Real-time Accurate Circle Fitting (RACF) algorithm implementation.

#     Parameters:
#         points (ndarray): Array of shape (N, 2) containing the x, y coordinates of the points.

#     Returns:
#         (float, float, float): Estimated circle parameters (cx, cy, r) where
#         (cx, cy) is the center and r is the radius.
#     """
#     # Ensure points are in numpy array
#     points = np.asarray(points)

#     # Calculate the mean of the points
#     x_mean = np.mean(points[:, 0])
#     y_mean = np.mean(points[:, 1])

#     # Center the points by subtracting the mean
#     centered_points = points - np.array([x_mean, y_mean])

#     # Formulate the matrices for least-squares fitting
#     Z = np.hstack((
#         centered_points[:, 0:1] ** 2 + centered_points[:, 1:2] ** 2,
#         centered_points[:, 0:1],
#         centered_points[:, 1:2],
#         np.ones((points.shape[0], 1))
#     ))

#     # Solve the linear system Z.T * Z * a = 0 using SVD or least-squares
#     U, S, Vt = np.linalg.svd(Z, full_matrices=False)
#     a = Vt.T[:, -1]  # Last column of Vt gives the solution

#     # Recover circle parameters
#     c = -0.5 * a[1:3] / a[0]
#     radius = np.sqrt((c[0] ** 2 + c[1] ** 2) - (a[3] / a[0]))

#     # Shift back the center to original coordinates
#     c += np.array([x_mean, y_mean])

#     return np.array(c).astype(int), int(radius)

# def fit_circle(points):
#     # Define the function to minimize
#     def residuals(params, x, y):
#         xc, yc, r = params
#         return np.sqrt((x - xc)**2 + (y - yc)**2) - r

#     # Extract x and y coordinates
#     x = points[:, 0]
#     y = points[:, 1]

#     # Initial guess (mean of points for center, average distance for radius)
#     x_m, y_m = np.mean(x), np.mean(y)
#     r_guess = np.mean(np.sqrt((x - x_m)**2 + (y - y_m)**2))
#     initial_guess = [x_m, y_m, r_guess]

#     # Least squares optimization
#     result = least_squares(residuals, initial_guess, args=(x, y))
#     xc, yc, r = result.x
#     return np.array([xc, yc]).astype(int), int(r)

# def cost_function(r, line_x, line_y, contour):
#     """
#     Compute the cost for a given radius r.
#     """
#     radius = r[0]
#     # Update cx and cy based on the given conditions
#     cx = line_x - radius if contour[:, 0].mean() < line_x else line_x + radius
#     cy = line_y - radius if contour[:, 1].mean() < line_y else line_y + radius

#     # Compute distances and enclosed ratio
#     distances = np.linalg.norm(contour - [cx, cy], axis=1)
#     outlier_ratio = np.mean(distances > radius)

#     # Compute cost
#     total_area = (2 * radius) ** 2  # Area (squared diameter)
#     cost = 1e7 * outlier_ratio
#     return cost

# def optimize_radius(line_x, line_y, contour, initial_guess):
#     """
#     Optimize the radius to minimize the cost function.
#     """
#     result = minimize(
#         cost_function, 
#         x0=[initial_guess],  # Initial guess for radius
#         args=(line_x, line_y, contour),  # Additional arguments for the cost function
#         bounds=[(0.1, None)]  # Radius must be positive
#     )
#     optimized_radius = result.x[0]
#     return optimized_radius, result.fun

# def min_enclosing_circle_tangent_to_lines2(contour, line_x, line_y):
#     """
#     Find the minimum enclosing circle of a contour that is tangent to both a vertical and a horizontal line.
    
#     Parameters:
#         contour (np.ndarray): Contour points as a numpy array of shape (n, 2).
#         line_x (float): X-coordinate of the vertical line.
#         line_y (float): Y-coordinate of the horizontal line.
    
#     Returns:
#         tuple: (center, radius) where center is (x, y) and radius is the circle radius.
#     """
#     # Find the initial minimum enclosing circle
#     (cx, cy), radius = cv2.minEnclosingCircle(contour)
#     # Adjust the x-coordinate of the circle center for vertical line tangency
#     if cx < line_x:
#         cx = line_x -radius
#     else:
#         cx = line_x + radius

#     # Adjust the y-coordinate of the circle center for horizontal line tangency
#     if cy < line_y:
#         cy = line_y -radius
#     else:
#         cy = line_y + radius
#     #optimize
#     # Recalculate the radius to ensure all points are enclosed
#     # distances = np.sqrt((contour[:, 0] - cx) ** 2 + (contour[:, 1] - cy) ** 2)
#     # within_circle = distances < radius
#     # enclosed_ratio = np.sum(within_circle) / within_circle.shape[0]
#     # total_area = radius*2
#     # cost = 10e5 * enclosed_ratio**2 + total_area
#     radius, f = optimize_radius(line_x, line_y, contour, radius)
#     if cx < line_x:
#         cx = line_x -radius
#     else:
#         cx = line_x + radius

#     # Adjust the y-coordinate of the circle center for horizontal line tangency
#     if cy < line_y:
#         cy = line_y -radius
#     else:
#         cy = line_y + radius
#     # Return the adjusted circle center and radius
#     return np.array((cx, cy)).astype(int), int(radius)

# def find_intersections(centers, y_top, y_bottom):
#     """
#     Calculate the intersection points of Line1 with horizontal lines at y_top and y_bottom.
    
#     Parameters:
#         centers (list of tuples): Two points [(x0, y0), (x1, y1)] defining Line1.
#         y_top (int): Y-coordinate of the horizontal line (Line2).
#         y_bottom (int): Y-coordinate of the horizontal line (Line3).
    
#     Returns:
#         tuple: Intersection points ((x_intersect1, y_top), (x_intersect2, y_bottom)).
#     """
#     # Extract points for Line1
#     (x0, y0), (x1, y1) = centers

#     # Calculate slope (m1) and intercept (c1) of Line1
#     if x1 != x0:  # Line1 is not vertical
#         m1 = (y1 - y0) / (x1 - x0)
#         c1 = y0 - m1 * x0
#     else:  # Line1 is vertical
#         m1 = float('inf')  # Infinite slope
#         c1 = x0  # Vertical line's x-coordinate

#     # Intersection with Line2 (y = y_top)
#     if m1 != float('inf'):
#         x_intersect1 = (y_top - c1) / m1
#     else:
#         x_intersect1 = c1  # For vertical Line1
#     intersection1 = (int(round(x_intersect1)), y_top)

#     # Intersection with Line3 (y = y_bottom)
#     if m1 != float('inf'):
#         x_intersect2 = (y_bottom - c1) / m1
#     else:
#         x_intersect2 = c1  # For vertical Line1
#     intersection2 = (int(round(x_intersect2)), y_bottom)

#     return np.asarray([intersection1, intersection2])


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
    # og_frame = cv2.imread("sample.jpg"); ret = True
    if ret:
        h, w, c = og_frame.shape
    else:
        print("read frame file")
        return None
    start_w = (w - 480) // 2
    end_w = start_w + 480
    
    crop_height = 200
    
    # Crop the middle section
    # og_frame = cv2.resize(og_frame, [640,480])
    og_frame = og_frame[crop_height:h-crop_height, start_w:end_w, :]
    t1 = time.perf_counter()
    results = model.predict(og_frame, show = True, verbose = False)
    
    t2 = time.perf_counter()
    if results[0].masks is None:
        print("No Studs detected")
        return None
    
    mask = results[0].masks.xy # this refers to all masks xy coordinates
    confs = results[0].boxes.conf
    conf = torch.min(results[0].boxes.conf).to('cpu').item()
    

    print(f"original mask length = {len(results[0].masks.xy)}")
    top_indices = np.where(confs > 0.6)[0]
    print(f"top indices = {top_indices}")
    best_masks = [results[0].masks.xy[i] for i in top_indices]
    print(f"best_masks length = {len(best_masks)}")

    if (len(results[0].masks) != 2): #only 2 studs are supported
        # print(f"{len(results[0].masks)} studs detected, expected 2. Abort computing offset")
        return None
    
    
    segments = np.zeros(og_frame.shape[:2])
    centers = []
    radiuses = []
    
    # Get the centers and radii of the all studs
    for i in range(len(mask)):
        (x, y), radius = cv2.minEnclosingCircle(mask[i])
        center = (int(x), int(y))  # Convert center coordinates to integers
        radius = int(radius)       # Convert radius to an integer
        centers.append(center)
        radiuses.append(radius)
        
    
    # find the top and bottom studs (compute median of all y values of each circle)
    top_stud_mask = mask[np.argmax([np.median(mask[0][:,1]), np.median(mask[1][:,1])])] #top stud is the one with higher Y value, which is at the BOTTOM in image
    bottom_stud_mask = mask[np.argmin([np.median(mask[0][:,1]), np.median(mask[1][:,1])])]
    
    
    y_top = np.min(top_stud_mask[:,1]).astype(int)
    y_bottom = np.max(bottom_stud_mask[:,1]).astype(int)
    
    # TODO: Chaitanya: Is the "200" a heuristic value? --> probably half of the width
    if(np.average([centers[0][0], centers[1][0]])< 200): # If average of the two centers is less than 200, so we are looking at the left side of the tool
        top_stud_side_x = np.max(top_stud_mask[:,0]).astype(int)
        bottom_stud_side_x = np.max(bottom_stud_mask[:,0]).astype(int)
    else:
        top_stud_side_x = np.min(top_stud_mask[:,0]).astype(int)
        bottom_stud_side_x = np.min(bottom_stud_mask[:,0]).astype(int)

    # from IPython import embed; embed()
    # intersects = find_intersections(centers, y_top, y_bottom)
    # target_center = np.average(intersects, axis=0)

    top_stud_center_adj, top_stud_radius_adj = min_enclosing_circle_tangent_to_lines(top_stud_mask, top_stud_side_x, y_top)
    bottom_stud_center_adj, bottom_stud_radius_adj = min_enclosing_circle_tangent_to_lines(bottom_stud_mask, bottom_stud_side_x, y_bottom)
    target_center_adj = np.mean([top_stud_center_adj, bottom_stud_center_adj], axis = 0)

    cv2.circle(og_frame, top_stud_center_adj,top_stud_radius_adj, [0,230,0], 2)
    cv2.circle(segments, top_stud_center_adj,top_stud_radius_adj, 150, 2)
    cv2.circle(og_frame, bottom_stud_center_adj,bottom_stud_radius_adj, [0,230,0], 2)
    cv2.circle(og_frame, target_center_adj.astype(np.int64), 6, [0,230,0], -1)
    cv2.imshow("og_frame", og_frame)
    cv2.waitKey(1)
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
    return np.concatenate([output[:2] / 1000, [output[2]]]) #mm to meter


if __name__ == "__main__":
    
    model = YOLO("studs-seg2.pt")
    camera = cv2.VideoCapture(2, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    camera.set(cv2.CAP_PROP_FPS, 10)

    filtered_center = np.array([0,0])

    # Set the loop rate (e.g., 10 Hz)
    while True:
        # Get the detected offset
        offset = compute_offset(camera, model)
        print(offset)
        # ret, og_frame = camera.read()
        # cv2.imshow("og_frame", og_frame)
        # cv2.waitKey(1)