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


# Convenience function to update locations of interest.
def update_locations(old:List[RealLocation], new:List[RealLocation]) -> None:
    '''Update locations with no duplicates.'''
    if new:
        for loc in new:
            if loc not in old:
                logging.getLogger('update_locations').info('New location of interest: {}'.format(loc))
                old.append(loc)

def main():
    # Initialize services
    cv_service = CVService(model_dir=CV_MODEL_DIR)
    nlp_service = MockNLPService(model_dir=NLP_MODEL_DIR)

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
    map_:SignedDistanceGrid = loc_service.get_map()
    # print("map_ jq print width", map_.width)
    # print("map_ jq print pixel", map_.grid[300][699])
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

    # Main loop

    
    while True:
        # Get new data
        pose, clues = loc_service.get_pose()
        # print("pose", pose)
        # print('clues id', clues[0][0], 'area of interest', clues[0][1])
        pose = pose_filter.update(pose)
        #print("pose2", pose)

        # capture image
        img = robot.camera.read_cv2_image(strategy='newest') #strategy is useless (unused)

        img = cv2.imread("/mnt/c/Users/user/Documents/GitHub/brainhack22_robotics/data/imgs/test_img.jpg")
    
        # return None

        if not pose:
            # If no pose
            continue
        
        #print('clues:', filter(new_clues, clues))
        # Filter out clues that were seen before
        clues = list(filter(new_clues, clues))

        # Process clues using NLP and determine any new locations of interest
        #print("Current lois:", test )
        
        if clues: # if there is new clue
            #print(clues)
            #In the case of new clues, extract their locations as new location of interest p
            new_lois = nlp_service.locations_from_clues(clues)
            # print("New", new_lois)
            # print("Old", lois)
            update_locations(lois, new_lois)

            #Record clues seen before
            seen_clues.update([c.clue_id for c in clues])

        # return None
        img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)

        targets = cv_service.targets_from_image(img)

        # Submit targets
        if targets:
            logging.getLogger('Main').info('{} targets detected.'.format(len(targets)))
            logging.getLogger('Reporting').info(rep_service.report(pose, img, targets))
        
        if not curr_loi:
            print(len(lois))
            if len(lois) == 0:
                logging.getLogger('Main').info('No more locations of interest.')
                # TODO: You ran out of LOIs. You could perform and random search for new
                # clues or targets

                while True:
                    # generate random pixel combination of grid coord
                    x = np.random.randint(0, map_.width/2)
                    y = np.random.randint(0, map_.height)

                    # check if LOI is an passable
                    if map_.grid[y][x] > 0:
                        break

                real_width = x * map_.scale
                real_height = y * map_.scale
                print("real width jq", real_width)
                print("real height jqq", real_height)

                curr_loi = RealLocation(real_width, real_height)
                logging.getLogger('Main').info('Current LOI set to: {}'.format(curr_loi))

                # Plan a path to the new LOI
                logging.getLogger('Main').info('Planning path to: {}'.format(curr_loi))
                
                path = planner.plan(pose[:2], curr_loi)
                path.reverse() # reverse so closest wp is last so that pop() is cheap
                
                curr_wp = None
                logging.getLogger('Main').info('Path planned.')

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
                    #Obtain the path step
                    curr_wp = path.pop()
                    logging.getLogger('Navigation').info('New waypoint: {}'.format(curr_wp))
                
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
                
                # TODO: Perform search behaviour? Participant to complete.\

                for i in range(8):
                    robot.chassis.drive_speed(z=22.5)
                    time.sleep(2.5)
                    robot.chassis.drive_speed(x=0, y=0, z=0)
                    time.sleep(2.5)
                    #img = robot.camera.read_cv2_image(strategy='newest') #strategy is useless (unused)
                    targets = cv_service.targets_from_image(img)
                    
                    # Submit targets
                    if targets:
                        logging.getLogger('Main').info('{} targets detected.'.format(len(targets)))
                        logging.getLogger('Reporting').info(rep_service.report(pose, img, targets))
                continue
        
    robot.chassis.drive_speed(x=0.0, y=0.0, z=0.0)  # set stop for safety
    logging.getLogger('Main').info('Mission Terminated.')


if __name__ == '__main__':
    main()