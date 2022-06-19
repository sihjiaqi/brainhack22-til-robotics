import logging
from re import X
from socket import timeout
from typing import List

from tilsdk import *                                            # import the SDK
# import optional useful things
from tilsdk.utilities import PIDController, SimpleMovingAverage
# Use this for the simulator
from tilsdk.mock_robomaster.robot import Robot
# from robomaster.robot import Robot                             # Use this for real robot

import time
# Import your code
from cv_service import CVService, MockCVService
from nlp_service import NLPService, MockNLPService
from planner import Planner
import matplotlib.pyplot as plt
import multiprocessing

# Setup logging in a nice readable format
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
                    datefmt='%H:%M:%S')

# Define config variables in an easily accessible location
# You may consider using a config file
REACHED_THRESHOLD_M = 0.25   # TODO: Participant may tune.
ANGLE_THRESHOLD_DEG = 20.0  # TODO: Participant may tune.
ROBOT_RADIUS_M = 0.25       # TODO: Participant may tune.
# TODO: Participant to fill in.
NLP_MODEL_DIR = "/mnt/c/Users/user/Documents/GitHub/brainhack22_robotics/model/nlp_model.onnx"
# TODO: Participant to fill in.
CV_MODEL_DIR = "/mnt/c/Users/user/Documents/GitHub/brainhack22_robotics/model/cv_onnx_model.onnx"


# Convenience function to update locations of interest.
def update_locations(old: List[RealLocation], new: List[RealLocation]) -> None:
    '''Update locations with no duplicates.'''
    if new:
        for loc in new:
            if loc not in old:
                logging.getLogger('update_locations').info(
                    'New location of interest: {}'.format(loc))
                old.append(loc)

def create_regions(width, height, region_size_w, region_size_h):

    region_width_size = int(region_size_w * width)
    region_height_size = int(region_size_h * height)
    

    x_regions = list(range(0, width+1, region_width_size))[1:]
    y_regions = list(range(0, height+1, region_height_size))[1:]

    if x_regions[-1] != width:
        x_regions.pop()
        x_regions.append(width)
    
    if y_regions[-1] != height:
        y_regions.pop()
        y_regions.append(height)

    print(x_regions)
    print(y_regions)

    x_q1 = region_size_w * width 
    x_q3 = (1-region_size_w) * width

    y_q1 = region_size_h * height 
    y_q3 = (1-region_size_h) * height

    # print(x_q1, x_q3)
    # print(y_q1, y_q3)

    regions_of_interest = []

    
    min_region_y = 0

    for y in y_regions:
        min_region_x = 0
        if y <= y_q1 or y > y_q3: 
            for x in x_regions:
                max_region = (x, y)
                min_region = (min_region_x, min_region_y)
                regions_of_interest.append((min_region, max_region))
                
                last_col_x_region = min_region_x    
                min_region_x = x 

        
        else:
            for x in x_regions:
                if x <= x_q1:
                    max_region = (x, y)
                    min_region = (min_region_x, min_region_y)

                    regions_of_interest.append((min_region, max_region))

                elif x > x_q3:
                    max_region = (x, y)
                    min_region = (last_col_x_region, min_region_y)

                    regions_of_interest.append((min_region, max_region))
                    
        min_region_y = y
                
    return regions_of_interest

def main():
    # Initialize services
    cv_service = CVService(model_dir=CV_MODEL_DIR)
    nlp_service = NLPService(model_dir=NLP_MODEL_DIR)

    # Input output
    # loc and rep should have the same port number
    loc_service = LocalizationService(host='localhost', port=5566)
    rep_service = ReportingService(host='localhost', port=5566)

    # initialize robot
    robot = Robot()
    robot.initialize(conn_type="sta")
    robot.camera.start_video_stream(display=False, resolution='720p')

    # start the run
    rep_service.start_run()

    # Initialize planner
    map_: SignedDistanceGrid = loc_service.get_map()
    # print("map_ jq print width", map_.width)
    # print("map_ jq print pixel", map_.grid[300][699])
    map_ = map_.dilated(1.5*ROBOT_RADIUS_M/map_.scale)
    planner = Planner(map_, sdf_weight=0.5)

    # Initialize variables and preset datatype (can be mixed)
    seen_clues = set()
    curr_loi: RealLocation = None   # Location Object
    # Contains a list of Locations Objects dtype
    path: List[RealLocation] = []
    # Contains a list of Locations Objects dtype
    lois: List[RealLocation] = []
    curr_wp: RealLocation = None    # Location Object
    step_counter = 0

    # Initialize tracker
    # TODO: Participant to tune PID controller values.
    # an instrument used in industrial control applications to regulate variables
    tracker = PIDController(Kp=(0.55, 0.55), Kd=(0.3, 0.3), Ki=(0.5, 0.5))

    # Initialize pose filter
    pose_filter = SimpleMovingAverage(n=5)

    # Define filter function to exclude clues seen before
    def new_clues(c): return c.clue_id not in seen_clues
    # print("new", new_clues)
    # print("seen", seen_clues)
    random_loi = False

    width = map_.width/2
    height = map_.height

    unexplored_list = create_regions(int(width), int(height), 0.33,0.25)
    # Main loop
    while True:
        # Get new data
        pose, clues = loc_service.get_pose()
        # print("pose", pose)
        # print('clues id', clues[0][0], 'area of interest', clues[0][1])
        pose = pose_filter.update(pose)
        #print("pose2", pose)

        # return None

        if not pose:
            # If no pose
            continue

        #print('clues:', filter(new_clues, clues))
        # Filter out clues that were seen before
        clues = list(filter(new_clues, clues))

        # Process clues using NLP and determine any new locations of interest
        #print("Current lois:", test )

        if clues:  # if there is new clue
            # print(clues)
            # In the case of new clues, extract their locations as new location of interest p
            new_lois = nlp_service.locations_from_clues(clues)
            # print("New", new_lois)
            # print("Old", lois)
            update_locations(lois, new_lois)
            print('lois new', new_lois)
            if new_lois:
                for new_loi in new_lois:
                    coord = (new_loi.x, new_loi.y)
                    print('coord', coord)
                    for i in unexplored_list:
                        if (coord[0] > i[0][0] and coord[0] < i[1][0]) and (coord[1] > i[0][1] and coord[1] < i[1][1]):
                            unexplored_list.remove(i)
                            print(i)

            # Record clues seen before
            seen_clues.update([c.clue_id for c in clues])

        if not curr_loi:
            print("Number of of LOIs", len(lois))
            if len(lois) == 0:
                logging.getLogger('Main').info('No more locations of interest.')
                # TODO: You ran out of LOIs. You could perform and random search for new
                # clues or targets

                print(len(unexplored_list))
                if len(unexplored_list) != 0:
                    unexplored_region = unexplored_list.pop(np.random.randint(0, len(unexplored_list)))
                    region_x_min = unexplored_region[0][0]
                    region_y_min = unexplored_region[0][1] 

                    region_x_max = unexplored_region[1][0]
                    region_y_max = unexplored_region[1][1] 
                    print(unexplored_region)
                    while True:
                        x = np.random.randint(region_x_min + 1, region_x_max)
                        y = np.random.randint(region_y_min + 1, region_y_max)
                        # check if LOI is an passable
                        if map_.grid[y][x] > 0:
                            break
                else:
                    while True:
                        # generate random pixel combination of grid coord
                        x = np.random.randint(0, width)
                        y = np.random.randint(0, height)

                        # check if LOI is an passable
                        if map_.grid[y][x] > 0:
                            break

                real_width = x * map_.scale
                real_height = y * map_.scale
                print("real width jq", real_width)
                print("real height jqq", real_height)

                curr_loi = RealLocation(real_width, real_height)
                logging.getLogger('Main').info(
                    'Current LOI set to: {}'.format(curr_loi))

                # Plan a path to the new LOI
                logging.getLogger('Main').info(
                    'Planning path to: {}'.format(curr_loi))

                path = planner.plan(pose[:2], curr_loi)
                path.reverse()  # reverse so closest wp is last so that pop() is cheap

                curr_wp = None
                logging.getLogger('Main').info('Path planned.')
                random_loi = True

            else:
                # Get new LOI
                lois.sort(key=lambda l: euclidean_distance(l, pose), reverse=True)
                curr_loi = lois.pop()

                logging.getLogger('Main').info(
                    'Current LOI set to: {}'.format(curr_loi))

                # Plan a path to the new LOI
                logging.getLogger('Main').info(
                    'Planning path to: {}'.format(curr_loi))

                path = planner.plan(pose[:2], curr_loi)
                #print('path', path)

                path.reverse()  # reverse so closest wp is last so that pop() is cheap

                # list_of_xs = []
                # list_of_ys = []
                # for i in path:
                #     list_of_xs.append(i.x)
                #     list_of_ys.append(i.y)
                # print(len(list_of_xs))
                # print(len(list_of_ys))
                # plt.plot(list_of_xs, list_of_ys)
                # plt.gca().invert_yaxis()
                # plt.show()
                curr_wp = None
                logging.getLogger('Main').info('Path planned.')
                

        else:
            # There is a current LOI objective.
            # Continue with navigation along current path.
            if path:
                # Get next waypoint
                if not curr_wp:
                    # Obtain the path step
                    curr_wp = path.pop()
                    logging.getLogger('Navigation').info('New waypoint: {}'.format(curr_wp))
                
                if len(lois) != 0 and random_loi == True:
                    curr_loi = None
                    logging.getLogger('Navigation').info('Location Importance Detected')
                    continue

                # Calculate distance and heading to waypoint
                dist_to_wp = euclidean_distance(pose, curr_wp)
                # print(dist_to_wp)

                ang_to_wp = np.degrees(np.arctan2(curr_wp[1]-pose[1], curr_wp[0]-pose[0]))
                # print(ang_to_wp)

                ang_diff = -(ang_to_wp - pose[2])  # body frame
                # print(ang_diff)

                # ensure ang_diff is in [-180, 180]
                if ang_diff < -180:
                    ang_diff += 360

                if ang_diff > 180:
                    ang_diff -= 360

                logging.getLogger('Navigation').debug('ang_to_wp: {}, hdg: {}, ang_diff: {}'.format(ang_to_wp, pose[2], ang_diff))

                # Consider waypoint reached if within a threshold distance
                if dist_to_wp < REACHED_THRESHOLD_M:
                    #print('Done liao')
                    logging.getLogger('Navigation').info('Reached wp: {}'.format(curr_wp))
                    tracker.reset()
                    curr_wp = None
                    continue

                # Determine velocity commands given distance and heading to waypoint
                #print(('dwp, ang', dist_to_wp, ang_diff))
                vel_cmd = tracker.update((dist_to_wp, ang_diff))
                #print('vel_cmd', vel_cmd)

                # return None

                # reduce x velocity
                vel_cmd[0] *= np.cos(np.radians(ang_diff))

                # If robot is facing the wrong direction, turn to face waypoint first before
                # moving forward.
                if abs(ang_diff) > ANGLE_THRESHOLD_DEG:
                    vel_cmd[0] = 0.0
                #print("abs(ang_diff) > ANGLE_THRESHOLD_DEG", abs(ang_diff) > ANGLE_THRESHOLD_DEG)

                # send command to robot to travel to the next waypoint
                robot.chassis.drive_speed(x=vel_cmd[0], z=vel_cmd[1])
                step_counter += 1
                print("steps taken by robot", step_counter)

                # capture images on every 100 steps
                if step_counter == 100:
                    print("robot moved 100 steps", step_counter)
                    # reset counter
                    step_counter = 0
                    # capture image
                    # img = robot.camera.read_cv2_image(strategy='newest') #strategy is useless (unused)
                    img = cv2.imread("/mnt/c/Users/user/Documents/GitHub/brainhack22_robotics/data/imgs/test_img.jpg")
                    img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
            
                    targets = cv_service.targets_from_image(img)

                    # submit targets
                    if targets:
                       logging.getLogger('Main').info('{} targets detected.'.format(len(targets)))
                       logging.getLogger('Reporting').info(rep_service.report(pose, img, targets))
                    
            else:
                logging.getLogger('Navigation').info('End of path.')
                curr_loi = None

                # TODO: Perform search behaviour? Participant to complete.
                for i in range(8):
                    robot.chassis.drive_speed(z=22.5)
                    time.sleep(2.5)
                    robot.chassis.drive_speed(x=0, y=0, z=0)
                    time.sleep(2.5)

                    # capture images when rotating
                    # img = robot.camera.read_cv2_image(strategy='newest') # strategy is useless (unused)
                    img = cv2.imread("/mnt/c/Users/user/Documents/GitHub/brainhack22_robotics/data/imgs/test_img.jpg")
                    img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
            
                    targets = cv_service.targets_from_image(img)

                    # submit targets
                    if targets:
                        logging.getLogger('Main').info('{} targets detected.'.format(len(targets)))
                        logging.getLogger('Reporting').info(rep_service.report(pose, img, targets))
                random_loi = False
                continue

    robot.chassis.drive_speed(x=0.0, y=0.0, z=0.0)  # set stop for safety
    logging.getLogger('Main').info('Mission Terminated.')


if __name__ == '__main__':
    main()
