import random
import shutil
from datetime import datetime
import pybullet as p
import time
import pybullet_data
import numpy as np
import sdf_optimization
import trimesh

# camera values
pitch = -90
roll = 0
upAxisIndex = 2
camDistance = 9.5
pixelWidth = 720
pixelHeight = 720
nearPlane = 0.01
farPlane = 100

fov = 20


# helper function which returns the direction for adjusting the pusher so it doesnt collide with the object
def sdf_radius_adjustment(direction):
    direction_adj = [0.0, 0.0, 0.0]
    if direction[0] >= 0:
        direction_adj[0] = -(finger_scale * 0.1)
    else:
        direction_adj[0] = (finger_scale * 0.1)
    if direction[1] >= 0:
        direction_adj[1] = -(finger_scale * 0.1)
    else:
        direction_adj[1] = (finger_scale * 0.1)
    return direction_adj


# sdf function needed for optimization
def sdf_function(finger_pos_sdf):
    return -trimesh.proximity.signed_distance(mesh, [finger_pos_sdf])[0]


# function to take an image of the goal region
def getImage(target):
    camTargetPos = target
    cameraUp = [0, 0, 1]
    cameraPos = target + [0, 0, 4]

    viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, 0, pitch,
                                                     roll, upAxisIndex)
    aspect = pixelWidth / pixelHeight
    projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
    img_arr = p.getCameraImage(pixelWidth,
                               pixelHeight,
                               viewMatrix,
                               projectionMatrix,
                               shadow=1,
                               lightDirection=[1, 1, 1],
                               renderer=p.ER_BULLET_HARDWARE_OPENGL)
    return img_arr


"""
this uses the optimization functions to calculate the optimal pushing position and adjusts the position regarding
the radius of the pusher
"""


def calculate_optimized_pos(finger_position, direction):
    found_optimized_pos, iterations, elapsed_time = sdf_optimization.optimize_push(finger_position, sdf_function,
                                                                                   (direction / 100), SDF_POS,
                                                                                   target_pos)
    temp_direction = np.asarray(target_pos) - np.asarray(found_optimized_pos)
    found_optimized_pos += sdf_radius_adjustment(temp_direction)

    print("optimized push position ", found_optimized_pos, "\n")
    return found_optimized_pos, iterations, elapsed_time


"""

I am printing all of the log messages, because I used pyCharm's functionality of saving console output to a text file
"""
if __name__ == "__main__":

    # current available and cleaned objects: bed, bottle, cabinet, chair, cup, lamp, monitor, sofa, table, vase
    for model in ["bed", "bottle", "cabinet", "chair", "cup", "lamp", "monitor", "sofa", "table", "vase"]:
        totalPushes = 0
        total_avg_iterations = 0.0
        total_avg_time = 0.0
        successful_runs = 0
        nr_runs = 100  # number of runs for each object

        obj_model = "../models-cleaned/experiments/obj/" + model + ".obj"
        file = open(r"../logs/push/logs.txt", "w")
        file.close()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time, "\n")

        for cur_run in range(1, nr_runs):
            print("RUN NUMBER:", cur_run)
            print(f"[{'=' * cur_run}>{'_' * (nr_runs - cur_run)}]")
            avg_iterations = 0
            avg_time = 0.0

            SDF_RAND_X = round(random.uniform(-4, 4), 1)
            SDF_RAND_Y = round(random.uniform(-4, 4), 1)
            SDF_RAND_Z = round(random.uniform(1, 2), 1)

            target_pos = np.asarray([round(random.uniform(-4, 4), 1), round(random.uniform(-4, 4), 1), 1])

            physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
            p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
            p.resetDebugVisualizerCamera(cameraDistance=7, cameraYaw=25, cameraPitch=-20,
                                         cameraTargetPosition=target_pos)
            # these are visual options which remove the gui and shadows
            # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            # p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

            p.setGravity(0, 0, -30)
            planeId = p.loadURDF("plane.urdf")

            print("selected obj", obj_model, "\n")
            # this extracts the scale of the object from its urdf file.
            # All of the urdf files are modified in the sense that line 12 is split right before the scale attribute
            with open(r"../models-cleaned/experiments/vhacd/urdf/" + model + ".urdf", 'r') as fp:
                # line to read
                line_number = 13
                # To store line
                lines = []
                for i, line in enumerate(fp, start=1):
                    # read line 13
                    if i == line_number:
                        lines.append(line.strip())

            object_scale = int(float(lines[0].split()[1]))
            print("scale of object: ", object_scale)

            mesh = trimesh.load(obj_model, force="mesh")
            mesh.apply_scale(object_scale)
            target_pos[2] = mesh.centroid[2]

            goal_pos = np.copy(target_pos)
            goal_pos[2] = -0.49
            sdf_goalId = p.loadURDF("util/goal_cylinder_push.urdf", goal_pos)
            sdf_goal_con = p.createConstraint(sdf_goalId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], goal_pos)

            sdfId = p.loadURDF("../models-cleaned/experiments/vhacd/urdf/" + model + ".urdf",
                               [SDF_RAND_X, SDF_RAND_Y, SDF_RAND_Z])

            print("sdf starting position: ", p.getBasePositionAndOrientation(sdfId)[0], "\n")

            time.sleep(.1)
            # simulate gravity for sdf
            for i in range(500):
                p.stepSimulation()
                time.sleep(1. / 360.)

            getImage(target_pos)

            SDF_POS = np.asarray([
                round(p.getBasePositionAndOrientation(sdfId)[0][0], 1),
                round(p.getBasePositionAndOrientation(sdfId)[0][1], 1),
                round(p.getBasePositionAndOrientation(sdfId)[0][2], 1)
            ])

            print("sdf actual position: ", SDF_POS, "\n")
            print("target position: ", target_pos, "\n")
            mesh.apply_translation(SDF_POS)

            target_direction = np.asarray(target_pos) - np.asarray(SDF_POS)
            target_direction[2] = mesh.centroid[2]
            print("target pushing direction: ", target_direction, "\n")

            sdf_orn = [0, 0, 0, 0]
            finger_scale = 0.5  # radius of sphere = 0.1 * finger_scale

            # q - (p-q / ||p-q|| * 1.2)
            finger_pos = SDF_POS - (((target_pos - SDF_POS) / (np.linalg.norm(target_pos - SDF_POS))) * 0.3)
            finger_pos[2] = mesh.center_mass[2]

            # pre-manipulate the pusher so that the object is in between the goal region and itself
            while sdf_function(finger_pos) <= 0.8 or np.linalg.norm(target_pos - SDF_POS) >= \
                    np.linalg.norm(target_pos - finger_pos):
                finger_pos -= (((target_pos - SDF_POS) / (np.linalg.norm(target_pos - SDF_POS))) * 0.1)
                finger_pos[2] = mesh.center_mass[2]

            finger_pos[2] = mesh.center_mass[2]
            print("position of finger after pre-manipulation: ", finger_pos, "\n")

            RIGHT_FINGER_POS = finger_pos
            right_finger = p.loadURDF("util/pushing_finger1.urdf", RIGHT_FINGER_POS, globalScaling=finger_scale)
            right_finger_con = p.createConstraint(right_finger, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                                  RIGHT_FINGER_POS)

            optimized_pos, iter_temp, time_temp = calculate_optimized_pos(RIGHT_FINGER_POS, target_direction)
            avg_iterations += iter_temp
            avg_time += time_temp

            x = RIGHT_FINGER_POS
            y = 0
            target_direction[2] = 0
            move_to_opt = True
            finished = False
            push_nr = 0
            while True:

                # check if the center of the object is currently in the goal region
                if np.linalg.norm(np.delete(p.getBasePositionAndOrientation(sdfId)[0] - target_pos, 2, axis=0)) <= .5 \
                        and not finished and push_nr > 0:
                    getImage(target_pos)
                    print("hey we made it")
                    print("this took", push_nr, "push attempts \n")
                    print("this took an average of", avg_iterations / push_nr, "iterations and an average of",
                          avg_time / push_nr, "seconds \n")

                    total_avg_time += avg_time / push_nr
                    total_avg_iterations += avg_iterations / push_nr
                    totalPushes += push_nr
                    successful_runs += 1
                    finished = True
                    time.sleep(1)
                    break

                elif push_nr >= 15:
                    print("We sadly failed")
                    finished = True
                    time.sleep(1)
                    break
                # in case our push was not successful we reposition and try again as long as we are within 15 attempts
                elif not move_to_opt and not finished:
                    # not optimal, because if the pusher is currently under the object like the table,
                    # it would just lift it and usually flip it over
                    x = p.getBasePositionAndOrientation(right_finger)[0]
                    g = x + np.asarray([0, 0, 3])

                    for i in range(200):
                        y = x + 1 / 2 * (1 - np.cos(np.pi * i / 300)) * (g - x)
                        p.changeConstraint(right_finger_con, y)
                        p.stepSimulation()
                        time.sleep(1. / 240.)

                    mesh = trimesh.load(obj_model, force="mesh")
                    mesh.apply_scale(object_scale)
                    p.resetBaseVelocity(sdfId, [0, 0, 0], [0, 0, 0])
                    SDF_POS, sdf_orn = p.getBasePositionAndOrientation(sdfId)
                    x, y, z, w = np.copy(sdf_orn)

                    M = [
                        [w * w + x * x - y * y - z * z, 2 * (-w * z + x * y), 2 * (w * y + x * z), 0],
                        [2 * (w * z + x * y), w * w - x * x + y * y - z * z, 2 * (-w * x + y * z), 0],
                        [2 * (-w * y + x * z), 2 * (w * x + y * z), w * w - x * x - y * y + z * z, 0],
                        [0, 0, 0, 1]
                    ]

                    mesh.apply_transform(M)
                    mesh.apply_translation(SDF_POS)
                    time.sleep(.1)
                    x = p.getBasePositionAndOrientation(right_finger)[0]

                    new_finger_pos = SDF_POS - (((target_pos - SDF_POS) / (np.linalg.norm(target_pos - SDF_POS))) * 0.5)

                    new_finger_pos[2] = mesh.center_mass[2]

                    # same pre-manipulation as before
                    while sdf_function(new_finger_pos) <= 0.8 or np.linalg.norm(target_pos - SDF_POS) >= np.linalg.norm(
                            target_pos - new_finger_pos):
                        new_finger_pos -= (((target_pos - SDF_POS) / (np.linalg.norm(target_pos - SDF_POS))) * 0.1)
                        new_finger_pos[2] = mesh.center_mass[2]

                    new_finger_pos[2] = x[2]

                    for k in range(200):
                        y = x + 1 / 2 * (1 - np.cos(np.pi * k / 200)) * (new_finger_pos - x)
                        p.changeConstraint(right_finger_con, y)
                        p.stepSimulation()
                        time.sleep(1. / 240.)

                    # splitting up the movement to the new pusher position so we are less likely to collide with the
                    # object
                    new_finger_pos[2] = random.uniform(finger_scale * 0.1, (SDF_POS[2] * 2) - 0.01)
                    x = p.getBasePositionAndOrientation(right_finger)[0]
                    for r in range(200):
                        y = x + 1 / 2 * (1 - np.cos(np.pi * r / 200)) * (new_finger_pos - x)
                        p.changeConstraint(right_finger_con, y)
                        p.stepSimulation()
                        time.sleep(1. / 240.)

                    x = p.getBasePositionAndOrientation(right_finger)[0]
                    target_direction = np.asarray(target_pos) - np.asarray(p.getBasePositionAndOrientation(sdfId)[0])
                    target_direction[2] = 0
                    optimized_pos, iter_temp, time_temp = calculate_optimized_pos(x, target_direction)
                    avg_iterations += iter_temp
                    avg_time += time_temp
                    move_to_opt = True

                # movement to optimized position
                if move_to_opt:
                    push_nr += 1
                    for i in range(200):
                        y = x + 1 / 2 * (1 - np.cos(np.pi * i / 200)) * (optimized_pos - x)
                        p.changeConstraint(right_finger_con, y)
                        p.stepSimulation()
                        time.sleep(1. / 240.)

                    f = p.getBasePositionAndOrientation(right_finger)[0]
                    # pushing the object to the goal position
                    for k in range(400):
                        # we break in the rare case that our object is already in the goal region so that we dont
                        # accidentally push it out of it again.
                        if np.linalg.norm(
                                np.delete(p.getBasePositionAndOrientation(sdfId)[0] - target_pos, 2, axis=0)) <= .25:
                            p.resetBaseVelocity(sdfId, [0, 0, 0], [0, 0, 0])
                            break
                        y = f + 1 / 2 * (1 - np.cos(np.pi * k / 400)) * \
                            (np.asarray(optimized_pos + target_direction) - f)
                        p.changeConstraint(right_finger_con, y)
                        p.stepSimulation()
                        time.sleep(1. / 240.)
                    move_to_opt = False

                p.stepSimulation()
                time.sleep(1. / 240.)

            p.disconnect()

        print(f"total Pushes: {totalPushes}")
        print(f"total Iterations: {total_avg_iterations}")
        print(f"total Average Time: {total_avg_time} \n")
        print("--" * 50, "\n")
        print(f"the pushing of the {model} took an average of, {totalPushes / successful_runs} push attempts over "
              f"{successful_runs} successful runs. \n")
        print(f"the calculation for the optimal position for the pushing of the {model} took an average of "
              f"{total_avg_iterations / successful_runs} iterations and an average of "
              f"{total_avg_time / successful_runs} seconds, over {successful_runs} successful runs. \n")
        print(f"We succeeded {successful_runs} times out of {nr_runs} runs. \n")
        print("--" * 50)
        time.sleep(1)
        # this will cause an error if there is no console output to log file set up
        shutil.copyfile(r"../logs/push/logs.txt", r"../logs/push/" + model + ".txt")
