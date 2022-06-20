import logging
from re import X
from socket import timeout
from typing import List

from tilsdk import *                                            # import the SDK
from tilsdk.utilities import PIDController, SimpleMovingAverage # import optional useful things
from tilsdk.mock_robomaster.robot import Robot                  # Use this for the simulator
#from robomaster.robot import Robot                             # Use this for real robot

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
NLP_MODEL_DIR = "/mnt/c/Users/user/Documents/GitHub/brainhack22_robotics/model/nlp_model.onnx"          # TODO: Participant to fill in.
CV_MODEL_DIR = "/mnt/c/Users/user/Documents/GitHub/brainhack22_robotics/model/cv_onnx_model.onnx"           # TODO: Participant to fill in.
#process_type = 'multiprocess'
process_type = 'sequence'
x_region_prop = 0.33
y_region_prop = 0.20

# Convenience function to update locations of interest.
def update_locations(old:List[RealLocation], new:List[RealLocation]) -> None:
    '''Update locations with no duplicates.'''
    if new:
        for loc in new:
            if loc not in old:
                logging.getLogger('update_locations').info('New location of interest: {}'.format(loc))
                old.append(loc)

# Seperation of map into important regions for bot to explore -------------------------------------------
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

    x_q1 = region_size_w * width 
    x_q3 = (1-region_size_w) * width
    y_q1 = region_size_h * height 
    y_q3 = (1-region_size_h) * height

    regions_of_interest = []

    min_region_y = 0

    # Creation of Regions
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
#-----------------------------------------------------------------------------------------------------------

# Main Code Launcher for differeny process types------------------------------------------------------------
def main(process_type='multiprocess'):

    # Initialization of Services
    loc_service = LocalizationService(host='localhost', port=5566)
    cv_service = CVService(model_dir=CV_MODEL_DIR)
    rep_service = ReportingService(host='localhost', port=5566)
    nlp_service = NLPService(model_dir=NLP_MODEL_DIR)

    # Initialization of Bot
    robot = Robot()
    #robot.initialize(conn_type="sta", sn="")
    robot.initialize(conn_type="sta")
    robot.camera.start_video_stream(display=False, resolution='720p')
    
    # Multiprocess Bot Functioning
    if process_type == 'multiprocess':    
        multiprocess_main(robot, loc_service, cv_service, rep_service, nlp_service)
    
    # Sequential Bot Function
    elif process_type == 'sequence':
        sequential_main(robot, loc_service, cv_service, rep_service, nlp_service)
    
    else:
        print('Please Check Spelling')

#-----------------------------------------------------------------------------------------------------------
    
# Main Source for Multiprocessing Functioning---------------------------------------------------------------
def multiprocess_main(robot, loc_service, cv_service, rep_service, nlp_service):
    global pose
    
    # Process 1- Handle bot movement (Requires: Robot, LOC and NLP Services)
    p1 = multiprocessing.Process(target = movement, args=(robot,loc_service, nlp_service ))

    # Process 2- Handle bot Image Capture and Prediction (Requires: Robot, LOC, CV and Rep Services)
    p2 = multiprocessing.Process(target = take_pic, args=(robot,loc_service, cv_service, rep_service ))

    p1.start()
    p2.start()
    p1.join()
    p2.join()
#-----------------------------------------------------------------------------------------------------------

# Process 2- Image Capture Process--------------------------------------------------------------------------
def take_pic(robot, loc_service, cv_service, rep_service):
    global pose
    
    # start the run
    rep_service.start_run()

    while True:
         # capture image
        img = robot.camera.read_cv2_image(strategy='newest') #strategy is useless (unused)
        
        pose, clues = loc_service.get_pose()
        img = cv2.imread("/mnt/c/Users/user/Documents/GitHub/brainhack22_robotics/data/imgs/test_img.jpg")
        img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)

        targets = cv_service.targets_from_image(img)
        #print(pose, img, targets)
        
        # Submit targets, pose and image
        if targets:
            logging.getLogger('Main').info('{} targets detected.'.format(len(targets)))
            logging.getLogger('Reporting').info(rep_service.report(pose, img, targets))

        time.sleep(2)
#-----------------------------------------------------------------------------------------------------------
            
# Process 1- Bot Path finding and Movement Handling---------------------------------------------------------
def movement(robot,loc_service, nlp_service):
    global pose 
    # Input output

    # Initialize planner
    map_:SignedDistanceGrid = loc_service.get_map()
    map_ = map_.dilated(1.5*ROBOT_RADIUS_M/map_.scale)
    planner = Planner(map_, sdf_weight=0.5)

    # Initialize variables and preset datatype (can be mixed)
    seen_clues = set() 
    curr_loi:RealLocation = None   # Location Object
    path:List[RealLocation] = []   # Contains a list of Locations Objects dtype
    lois:List[RealLocation] = []   # Contains a list of Locations Objects dtype
    curr_wp:RealLocation = None    # Location Object


    # Initialize tracker
    # TODO: Participant to tune PID controller values.
    tracker = PIDController(Kp=(0.55, 0.55), Kd=(0.3, 0.3), Ki=(0.5, 0.5)) #an instrument used in industrial control applications to regulate variables

    # Initialize pose filter
    pose_filter = SimpleMovingAverage(n=5) 

    # Define filter function to exclude clues seen before   
    new_clues = lambda c: c.clue_id not in seen_clues
    # print("new", new_clues)
    # print("seen", seen_clues)


    # Used for switching path ways in the case of location pick ups during random loi
    random_loi = False

    # Map Details 
    width = map_.width/2
    height = map_.height

    # List of Unexplored Regions (Will be filtered once been to)
    unexplored_list = create_regions(int(width), int(height), x_region_prop, y_region_prop )
    
    #Main Loop
    while True:
        # Get new data
        pose, clues = loc_service.get_pose()

        # print("pose", pose)
        # print('clues id', clues[0][0], 'area of interest', clues[0][1])

        pose = pose_filter.update(pose)
        #print("pose2", pose)

        if not pose:
            # If no pose
            continue

        # Filter out clues that were seen before
        clues = list(filter(new_clues, clues))

        # Process clues using NLP and determine any new locations of interest
        if clues: # if there is new clue
            #print(clues)
            #In the case of new clues, extract their locations as new location of interest p
            new_lois = nlp_service.locations_from_clues(clues)
            # print("New", new_lois)
            # print("Old", lois)
            update_locations(lois, new_lois)
            #print('lois new', new_lois)

            # Used to filter list of unexplored regions. If clue lead to here, filter it
            if new_lois:
                for new_loi in new_lois:
                    coord = (new_loi.x / map_.scale, new_loi.y / map_.scale)
                    print('coord', coord)
                    for i in unexplored_list:
                        if (coord[0] > i[0][0] and coord[0] <= i[1][0]) and (coord[1] > i[0][1] and coord[1] <= i[1][1]):
                            unexplored_list.remove(i)
            
            #Record clues seen before
            seen_clues.update([c.clue_id for c in clues])
        
        if not curr_loi:
            if len(lois) == 0:
                logging.getLogger('Main').info('No more locations of interest.')
                # TODO: You ran out of LOIs. You could perform and random search for new clues or targets

                # Contains 2 types of random search (Priotity Random and Random)

                # Random Search --------------------------------------------------------------------------------------

                # Picks random region from list of unexplored regions (Priority random)
                if len(unexplored_list) != 0:
                    unexplored_region = unexplored_list.pop(np.random.randint(0, len(unexplored_list)))
                    region_x_min = unexplored_region[0][0]
                    region_y_min = unexplored_region[0][1] 

                    region_x_max = unexplored_region[1][0]
                    region_y_max = unexplored_region[1][1]

                    #print(region_x_min, region_x_max)
                    #print(region_y_min, region_y_max) 
                    #print(unexplored_region)

                    # Select Center of random unexplored region
                    x = int(region_x_min + ((region_x_max - region_x_min) /2))
                    y = int(region_y_min + ((region_y_max - region_y_min) /2))

                    # In the case of unavailablity, select a random location within the region till available
                    if map_.grid[y][x] < 0:
                        unavailabilty_count = 0
                        while True:
                            x = np.random.randint(region_x_min + 1, region_x_max)
                            y = np.random.randint(region_y_min + 1, region_y_max)
                            # check if LOI is an passable
                            if map_.grid[y][x] > 0:
                                break

                            #In the case of supreme unavailability select random 
                            if unavailabilty_count > 200:
                                while True:
                                    # generate random pixel combination of grid coord
                                    unexplored_region = 0
                                    x = np.random.randint(0, width)
                                    y = np.random.randint(0, height)

                                    # check if LOI is an passable
                                    if map_.grid[y][x] > 0:
                                        break
                            unavailabilty_count += 1       
                else:
                    # If no more unexplored regions, select completely randomly (Random)
                    while True:
                        # generate random pixel combination of grid coord
                        x = np.random.randint(0, width)
                        y = np.random.randint(0, height)

                        # check if LOI is an passable
                        if map_.grid[y][x] > 0:
                            break
                #print(len(unexplored_list))
                #print(unexplored_list)

                real_width = x * map_.scale
                real_height = y * map_.scale
                # print("real width jq", real_width)
                # print("real height jqq", real_height)

                curr_loi = RealLocation(real_width, real_height)
                logging.getLogger('Main').info('Current LOI set to: {}'.format(curr_loi))

                # Plan a path to the new LOI
                logging.getLogger('Main').info('Planning path to: {}'.format(curr_loi))
                
                path = planner.plan(pose[:2], curr_loi)
                path.reverse() # reverse so closest wp is last so that pop() is cheap
                
                curr_wp = None
                logging.getLogger('Main').info('Path planned.')

                # Track if current coord is randomly selected
                random_loi = True
                #------------------------------------------------------------------------------------------------------

            else:
                # Get new LOI
                lois.sort(key=lambda l: euclidean_distance(l, pose), reverse=True)
                
                curr_loi = lois.pop()

                
                logging.getLogger('Main').info('Current LOI set to: {}'.format(curr_loi))

                # Plan a path to the new LOI
                logging.getLogger('Main').info('Planning path to: {}'.format(curr_loi))
                
                path = planner.plan(pose[:2], curr_loi)
                #print('path', path)
                
                path.reverse() # reverse so closest wp is last so that pop() is cheap
                
                curr_wp = None
                logging.getLogger('Main').info('Path planned.')
            
        else:
            # There is a current LOI objective.
            # Continue with navigation along current path.
            if path:
                # Get next waypoint
                if not curr_wp:
                    #Obtain the path step
                    curr_wp = path.pop()
                    logging.getLogger('Navigation').info('New waypoint: {}'.format(curr_wp))

                # In the case where there are better places to explore from clues and that the bot is currently on random path
                if len(lois) != 0 and random_loi == True:
                    curr_loi = None

                    #Shift Prioirity to clue location
                    logging.getLogger('Navigation').info('Location Importance Detected')

                    # If I was moving to a Priority Random Location, put it back to list as it may have more stuff to explore there. 
                    if unexplored_region != 0:
                        unexplored_list.append(unexplored_region)
                        unexplored_region = 0

                    continue
                
                # Calculate distance and heading to waypoint

                dist_to_wp = euclidean_distance(pose, curr_wp)
                #print(dist_to_wp)

                
                ang_to_wp = np.degrees(np.arctan2(curr_wp[1]-pose[1], curr_wp[0]-pose[0]))
                #print(ang_to_wp)
                
                ang_diff = -(ang_to_wp - pose[2]) # body frame
                #print(ang_diff)
                

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

                #return None

                # reduce x velocity
                vel_cmd[0] *= np.cos(np.radians(ang_diff))
                
                # If robot is facing the wrong direction, turn to face waypoint first before
                # moving forward.
                if abs(ang_diff) > ANGLE_THRESHOLD_DEG:
                    vel_cmd[0] = 0.0
                #print("abs(ang_diff) > ANGLE_THRESHOLD_DEG", abs(ang_diff) > ANGLE_THRESHOLD_DEG)

                
                # Send command to robot
                robot.chassis.drive_speed(x=vel_cmd[0], z=vel_cmd[1])
                
            else:
                logging.getLogger('Navigation').info('End of path.')
                curr_loi = None
                #return None
                
                # TODO: Perform search behaviour? Participant to complete.

                # 360 Picture Taking ----------------------------------------------------------------------------------

                for i in range(8):
                    robot.chassis.drive_speed(z=22.5)
                    time.sleep(2.5)
                    robot.chassis.drive_speed(x=0, y=0, z=0)
                    time.sleep(2.5)
                    #img = robot.camera.read_cv2_image(strategy='newest') #strategy is useless (unused)
                    #targets = cv_service.targets_from_image(img)
                    
                    # Submit targets
                    # if targets:
                    #     logging.getLogger('Main').info('{} targets detected.'.format(len(targets)))
                    #     logging.getLogger('Reporting').info(rep_service.report(pose, img, targets))
                random_loi = False
                continue
        
    robot.chassis.drive_speed(x=0.0, y=0.0, z=0.0)  # set stop for safety
    logging.getLogger('Main').info('Mission Terminated.')
#-----------------------------------------------------------------------------------------------------------

# Sequential Coded Bot--------------------------------------------------------------------------------------
def sequential_main(robot, loc_service, cv_service, rep_service, nlp_service):

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
    # Used for switching path ways in the case of location pick ups during random loi
    random_loi = False
    
    # Map details
    width = map_.width/2
    height = map_.height

    unexplored_list = create_regions(int(width), int(height), x_region_prop, y_region_prop)
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
            #print('lois new', new_lois)

            # Used to filter list of unexplored regions. If clue lead to here, filter it
            if new_lois:
                for new_loi in new_lois:
                    coord = (new_loi.x/map_.scale, new_loi.y/map_.scale)
                    #print('coord', coord)
                    
                    for i in unexplored_list:
                        if (coord[0] > i[0][0] and coord[0] <= i[1][0]) and (coord[1] > i[0][1] and coord[1] <= i[1][1]):
                            unexplored_list.remove(i)
                            print(i)

            # Record clues seen before
            seen_clues.update([c.clue_id for c in clues])

        if not curr_loi:
            print("Number of of LOIs", len(lois))
            if len(lois) == 0:
                logging.getLogger('Main').info('No more locations of interest.')
                # TODO: You ran out of LOIs. You could perform and random search for new clues or targets

                # Contains 2 types of random search (Priotity Random and Random)

                # Random Search --------------------------------------------------------------------------------------

                # Picks random region from list of unexplored regions (Priority random)

                if len(unexplored_list) != 0:
                    unexplored_region = unexplored_list.pop(np.random.randint(0, len(unexplored_list)))
                    region_x_min = unexplored_region[0][0]
                    region_y_min = unexplored_region[0][1] 

                    region_x_max = unexplored_region[1][0]
                    region_y_max = unexplored_region[1][1] 
                    # print(region_x_min, region_x_max)
                    # print(region_y_min, region_y_max) 

                    # print(unexplored_region)
                    x = int(region_x_min + ((region_x_max - region_x_min) /2))
                    y = int(region_y_min + ((region_y_max - region_y_min) /2))
                
                    if map_.grid[y][x] < 0:

                        unavailabilty_count = 0
                        while True:
                            x = np.random.randint(region_x_min + 1, region_x_max)
                            y = np.random.randint(region_y_min + 1, region_y_max)
                            # check if LOI is an passable
                            if map_.grid[y][x] > 0:
                                break
                            
                            #In the case of supreme unavailability select random 
                            if unavailabilty_count > 200:
                                while True:
                                    # generate random pixel combination of grid coord
                                    unexplored_region = 0
                                    x = np.random.randint(0, width)
                                    y = np.random.randint(0, height)

                                    # check if LOI is an passable
                                    if map_.grid[y][x] > 0:
                                        break
                            unavailabilty_count += 1       
                else:
                    while True:
                        # generate random pixel combination of grid coord
                        x = np.random.randint(0, width)
                        y = np.random.randint(0, height)

                        # check if LOI is an passable
                        if map_.grid[y][x] > 0:
                            break

                # print(len(unexplored_list))
                # print(unexplored_list)

                real_width = x * map_.scale
                real_height = y * map_.scale


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
                #-----------------------------------------------------------------------------------------------------

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
    
                # reverse so closest wp is last so that pop() is cheap
                path.reverse()  
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

                # In the case where there are better places to explore from clues and that the bot is currently on random path
                if len(lois) != 0 and random_loi == True:
                    curr_loi = None
                    logging.getLogger('Navigation').info('Location Importance Detected')

                    # If I was moving to a Priority Random Location, put it back to list as it may have more stuff to explore there. 
                    if unexplored_region != 0:
                        unexplored_list.append(unexplored_region)
                        unexplored_region = 0
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
                #print("steps taken by robot", step_counter)

                # capture images on every 100 steps
                if step_counter == 100:
                    #print("robot moved 100 steps", step_counter)
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

                # 360 Picture Taking ----------------------------------------------------------------------------------
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
    main(process_type)