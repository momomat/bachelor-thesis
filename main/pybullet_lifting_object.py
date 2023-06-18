import random
import shutil
from datetime import datetime
import pybullet as p
import time
import pybullet_data
import numpy as np
import sdf_optimization
import trimesh


# helper function
def sdf_radius_adjustment(direction):
    direction_adj = [0.0, 0.0, 0.0]
    if direction[0] >= 0:
        direction_adj[0] = -0.1
    else:
        direction_adj[0] = 0.1
    if direction[1] >= 0:
        direction_adj[1] = -0.1
    else:
        direction_adj[1] = 0.1
    return direction_adj


def sdf_function(finger_pos_sdf):
    return -trimesh.proximity.signed_distance(mesh, [finger_pos_sdf])[0]

def calculate_optimized_pos(random_vector):
    optimized_positions, iter_temp, time_temp = \
        sdf_optimization.optimize_grasp(np.concatenate([LEFT_FINGER_POS, RIGHT_FINGER_POS]), sdf_function)

    optimized_l = optimized_positions[:3]
    optimized_r = optimized_positions[-3:]

    # adjust for the radius of the grasping spheres, dependant on the previous calculated random vector
    optimized_l -= random_vector * (finger_scale * 0.1)
    optimized_r += random_vector * (finger_scale * 0.1)

    print("optimized grasp position for left finger ", optimized_pos_left)
    print("optimized grasp position for right finger ", optimized_pos_right)
    return optimized_l, optimized_r, iter_temp, time_temp

if __name__ == "__main__":
    # current available and cleaned + vhacd objects: bed, bottle, cabinet, chair, cup, lamp, monitor, sofa, table, vase
    for model in ["bed", "bottle", "cabinet", "chair", "cup", "lamp", "monitor", "sofa", "table", "vase"]:
        totalGrasps = 0
        total_avg_iterations = 0.0
        total_avg_time = 0.0
        nr_runs = 100 # number of runs for each object
        successful_runs = 0

        obj_model = "../models-cleaned/experiments/obj/" + model + ".obj"

        # for this to be useful you have to setup console output to the specified log file
        file = open(r"../logs/grasp/logs.txt", "w")
        file.close()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time, "\n")

        for cur_run in range(1, nr_runs):
            print("RUN NUMBER:", cur_run)
            print(f"[{'=' * cur_run}>{'_' * (nr_runs - cur_run)}]")
            avg_iterations = 0
            avg_time = 0.0
            grasp_nr = 0

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

            p.setGravity(0, 0, -10)
            planeId = p.loadURDF("plane.urdf")

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

            sdf_goalId = p.loadURDF("util/goal_cylinder_grasp.urdf", [target_pos[0], target_pos[1], 0.25])
            sdf_goal_con = p.createConstraint(sdf_goalId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [target_pos[0], target_pos[1], 0.25])

            sdfId = p.loadURDF("../models-cleaned/experiments/vhacd/urdf/" + model + ".urdf", [SDF_RAND_X, SDF_RAND_Y, SDF_RAND_Z])
            print("selected obj", obj_model)

            print("sdf starting position: ", p.getBasePositionAndOrientation(sdfId)[0], "\n")
            print("target position", target_pos)

            # simulate gravity for sdf
            for i in range(500):
                p.stepSimulation()
                time.sleep(1. / 240.)

            SDF_POS = np.asarray([
                round(p.getBasePositionAndOrientation(sdfId)[0][0], 1),
                round(p.getBasePositionAndOrientation(sdfId)[0][1], 1),
                round(p.getBasePositionAndOrientation(sdfId)[0][2], 1)
            ])
            print("sdf actual position: ", SDF_POS, "\n")
            mesh.apply_translation(SDF_POS)
            object_height = SDF_POS[2] * 2

            finger_pos = SDF_POS
            finger_pos[2] = mesh.center_mass[2] - 0.05

            # pre-manipulate the grasping spheres by placing them on a randomized vector
            mu, sigma = 0, 0.1  # mean and standard deviation
            s = np.random.normal(mu, sigma, (3,))
            norm = s / np.linalg.norm(s)
            norm[2] = 0
            LEFT_FINGER_POS = np.copy(finger_pos) - norm
            RIGHT_FINGER_POS = np.copy(finger_pos) + norm

            # pre-manipulate the grasping spheres so that the object is in between the two spheres
            while sdf_function(LEFT_FINGER_POS) <= 0.1 or sdf_function(RIGHT_FINGER_POS) <= 0.1:
                LEFT_FINGER_POS -= (norm * 1.5)
                RIGHT_FINGER_POS += (norm * 1.5)

            print("position of fingers after pre-manipulation: ", finger_pos, "\n")
            finger_scale = 0.5

            left_finger = p.loadURDF("util/finger1.urdf", LEFT_FINGER_POS, globalScaling=finger_scale)
            right_finger = p.loadURDF("util/finger2.urdf", RIGHT_FINGER_POS, globalScaling=finger_scale)

            left_finger_con = p.createConstraint(left_finger, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], LEFT_FINGER_POS)
            right_finger_con = p.createConstraint(right_finger, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                                  RIGHT_FINGER_POS)


            # adjust for the radius of the grasping spheres, dependant on the previous calculated random vector
            optimized_pos_left, optimized_pos_right = calculate_optimized_pos(norm)



            orn = p.getQuaternionFromEuler([0, 0, 0])
            x_right = RIGHT_FINGER_POS
            x_left = LEFT_FINGER_POS
            y_right = 0
            y_left = 0

            move_to_opt = True
            finished = False
            reGrasp = False
            while True:

                # moving to the calculated position and starting our grasp
                if move_to_opt and not reGrasp and not finished:
                    # making sure we are within the 15 grasping attempts
                    if grasp_nr >= 15:
                        print("We sadly failed")
                        finished = True
                        time.sleep(1)
                        break
                    grasp_nr += 1
                    x_right = np.copy(p.getBasePositionAndOrientation(right_finger)[0])
                    x_left = np.copy(p.getBasePositionAndOrientation(left_finger)[0])
                    for i in range(200):
                        if i <= 200:
                            y_right = x_right + 1 / 2 * (1 - np.cos(np.pi * i / 200)) * (optimized_pos_right - x_right)
                            y_left = x_left + 1 / 2 * (1 - np.cos(np.pi * i / 200)) * (optimized_pos_left - x_left)

                            p.changeConstraint(right_finger_con, y_right)
                            p.changeConstraint(left_finger_con, y_left)

                            p.stepSimulation()
                            time.sleep(1. / 240.)

                    r_old = np.copy(p.getBasePositionAndOrientation(right_finger)[0])
                    l_old = np.copy(p.getBasePositionAndOrientation(left_finger)[0])
                    goal_r = np.copy(p.getBasePositionAndOrientation(left_finger)[0])
                    goal_l = np.copy(p.getBasePositionAndOrientation(right_finger)[0])

                    # stabilize the object between the grasping spheres before starting the actual lift
                    for f in range(400):
                        if f <= 400:
                            y_right = r_old + 1 / 2 * (1 - np.cos(np.pi * f / 400)) * (goal_r - r_old)
                            y_left = l_old + 1 / 2 * (1 - np.cos(np.pi * f / 400)) * (goal_l - l_old)

                            p.changeConstraint(right_finger_con, y_right)
                            p.changeConstraint(left_finger_con, y_left)

                            p.stepSimulation()
                            time.sleep(1. / 240.)

                            goal_l = y_right
                            goal_r = y_left

                    f_right = np.copy(p.getBasePositionAndOrientation(right_finger)[0])
                    f_left = np.copy(p.getBasePositionAndOrientation(left_finger)[0])

                    optimized_lift_l = np.copy(p.getBasePositionAndOrientation(sdfId)[0]) + [0, 0, 2]
                    optimized_lift_r = np.copy(p.getBasePositionAndOrientation(sdfId)[0]) + [0, 0, 2]

                    # apply more pressure so object doesnt keep hanging
                    optimized_lift_l[:2] = np.copy(target_pos[:2]) + (
                                (f_left[:2] - np.asarray(p.getBasePositionAndOrientation(sdfId)[0][:2])) * 0.6)
                    optimized_lift_r[:2] = np.copy(target_pos[:2]) + (
                                (f_right[:2] - np.asarray(p.getBasePositionAndOrientation(sdfId)[0][:2])) * 0.6)
                    t_right = 0
                    t_left = 0

                    for n in range(1000):  # should probably scale the max range depending on distance
                        if np.linalg.norm(p.getBasePositionAndOrientation(sdfId)[0][:2] - target_pos[:2]) <= .25:
                            print("we done early:)")
                            move_to_opt = False
                            break

                            # updating the mesh's orientation and position to ensure the sdf function stays accurate
                        if n % 100 == 0:
                            mesh = trimesh.load(obj_model, force="mesh")
                            mesh.apply_scale(object_scale)

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

                            # checking if the object is still being grasped
                            if sdf_function(p.getBasePositionAndOrientation(left_finger)[0]) >= \
                                    (0.15 + (finger_scale * 0.1)):
                                reGrasp = True
                                break
                        # this just ensures more stability at the start of the grasp so that the object doesnt slip
                        if n <= 100:
                            t_right = f_right + 1 / 2 * (1 - np.cos(np.pi * n / 1000)) * (optimized_lift_l - f_right)
                            t_left = f_left + 1 / 2 * (1 - np.cos(np.pi * n / 1000)) * (optimized_lift_r - f_left)
                        if 100 <= n <= 1000:
                            t_right = f_right + 1 / 2 * (1 - np.cos(np.pi * n / 1000)) * (optimized_lift_r - f_right)
                            t_left = f_left + 1 / 2 * (1 - np.cos(np.pi * n / 1000)) * (optimized_lift_l - f_left)

                        p.changeConstraint(right_finger_con, t_right)
                        p.changeConstraint(left_finger_con, t_left)

                        p.stepSimulation()
                        time.sleep(1. / 240.)

                    move_to_opt = False

                # we are regrasping in case the object slipped and escaped our grasp
                elif reGrasp and not finished:
                    print("regrasping...")
                    p.resetBaseVelocity(sdfId, [0, 0, 0], [0, 0, 0])

                    finger_l_pos = p.getBasePositionAndOrientation(left_finger)[0]
                    finger_r_pos = p.getBasePositionAndOrientation(right_finger)[0]
                    l_goal = finger_l_pos + np.asarray([0, 0, 2])
                    r_goal = finger_r_pos + np.asarray([0, 0, 2])
                    for k in range(400):
                        y_left = finger_l_pos + 1 / 2 * (1 - np.cos(np.pi * k / 400)) * (l_goal - finger_l_pos)
                        y_right = finger_r_pos + 1 / 2 * (1 - np.cos(np.pi * k / 400)) * (r_goal - finger_r_pos)

                        p.changeConstraint(left_finger_con, y_left)
                        p.changeConstraint(right_finger_con, y_right)
                        p.stepSimulation()
                        time.sleep(1. / 240.)

                    p.resetBaseVelocity(sdfId, [0, 0, 0], [0, 0, 0])

                    mesh = trimesh.load(obj_model, force="mesh")
                    mesh.apply_scale(object_scale)
                    SDF_POS, sdf_orn = p.getBasePositionAndOrientation(sdfId)
                    x, y, z, w = np.copy(sdf_orn)

                    M = [
                        [w * w + x * x - y * y - z * z, 2 * (-w * z + x * y), 2 * (w * y + x * z), 0],
                        [2 * (w * z + x * y), w * w - x * x + y * y - z * z, 2 * (-w * x + y * z), 0],
                        [2 * (-w * y + x * z), 2 * (w * x + y * z), w * w - x * x - y * y + z * z, 0],
                        [0, 0, 0, 1]
                    ]
                    mesh.apply_transform(M)
                    mesh.apply_translation(SDF_POS)  # dont forget orientation incase object falls down

                    finger_pos = np.asarray(p.getBasePositionAndOrientation(sdfId)[0])
                    finger_pos[2] = mesh.center_mass[2] - 0.05

                    # pre-manipulate the grasping spheres like before by placing them on a new randomized vector
                    mu, sigma = 0, 0.1  # mean and standard deviation
                    s = np.random.normal(mu, sigma, (3,))
                    norm = s / np.linalg.norm(s)
                    norm[2] = 0
                    regrasp_left = np.copy(finger_pos) - norm
                    regrasp_right = np.copy(finger_pos) + norm

                    while sdf_function(regrasp_left) <= 0.5 or sdf_function(regrasp_right) <= 0.5:
                        regrasp_left -= (norm * 1.5)
                        regrasp_right += (norm * 1.5)

                    finger_l_pos = np.asarray(p.getBasePositionAndOrientation(left_finger)[0])
                    finger_r_pos = np.asarray(p.getBasePositionAndOrientation(right_finger)[0])

                    for k in range(700):

                        y_left = finger_l_pos + 1 / 2 * (1 - np.cos(np.pi * k / 700)) * (regrasp_left - finger_l_pos)
                        y_right = finger_r_pos + 1 / 2 * (1 - np.cos(np.pi * k / 700)) * (regrasp_right - finger_r_pos)

                        p.changeConstraint(left_finger_con, y_left)
                        p.changeConstraint(right_finger_con, y_right)
                        p.stepSimulation()
                        time.sleep(1. / 240.)
                    mesh = trimesh.load(obj_model, force="mesh")
                    mesh.apply_scale(object_scale)

                    SDF_POS, sdf_orn = p.getBasePositionAndOrientation(sdfId)
                    x, y, z, w = np.copy(sdf_orn)

                    M = [
                        [w * w + x * x - y * y - z * z, 2 * (-w * z + x * y), 2 * (w * y + x * z), 0],
                        [2 * (w * z + x * y), w * w - x * x + y * y - z * z, 2 * (-w * x + y * z), 0],
                        [2 * (-w * y + x * z), 2 * (w * x + y * z), w * w - x * x - y * y + z * z, 0],
                        [0, 0, 0, 1]
                    ]
                    # update mesh in case the collided during re-grasping
                    mesh.apply_transform(M)
                    mesh.apply_translation(SDF_POS)

                    finger_l_pos = p.getBasePositionAndOrientation(left_finger)[0]
                    finger_r_pos = p.getBasePositionAndOrientation(right_finger)[0]

                    optimized_positions, iter_temp, time_temp = sdf_optimization.optimize_grasp(np.concatenate([finger_l_pos, finger_r_pos]),
                                                                          sdf_function)
                    avg_iterations += iter_temp
                    avg_time += time_temp

                    optimized_pos_left = optimized_positions[:3]
                    optimized_pos_right = optimized_positions[-3:]

                    # adjust for the radius of the grasping spheres, dependant on the previous calculated random vector
                    optimized_pos_left -= norm * (finger_scale * 0.1)
                    optimized_pos_right += norm * (finger_scale * 0.1)
                    move_to_opt = True
                    reGrasp = False

                # check if the center of the object is currently above and inside of the goal region
                elif np.linalg.norm(np.delete(p.getBasePositionAndOrientation(sdfId)[0] - target_pos, 2, axis=0)) <= .75 \
                        and not finished:

                    print("hey we made it")
                    finished = True
                    for _ in range(500):
                        p.stepSimulation()
                        time.sleep(1. / 240.)

                    p.resetBaseVelocity(sdfId, [0, 0, 0], [0, 0, 0])

                    f_right = p.getBasePositionAndOrientation(right_finger)[0]
                    f_left = p.getBasePositionAndOrientation(left_finger)[0]
                    end_pos_r = np.copy(f_left)
                    end_pos_l = np.copy(f_right)
                    end_pos_l[-1:] = object_height + 0.4
                    end_pos_r[-1:] = object_height + 0.4
                    t_right = 0
                    t_left = 0
                    # attempt to set the object down on top of the platform
                    for r in range(400):
                        if r <= 400:
                            t_right = f_right + 1 / 2 * (1 - np.cos(np.pi * r / 400)) * (end_pos_r - f_right)
                            t_left = f_left + 1 / 2 * (1 - np.cos(np.pi * r / 400)) * (end_pos_l - f_left)

                        p.changeConstraint(right_finger_con, t_right)
                        p.changeConstraint(left_finger_con, t_left)

                        p.stepSimulation()
                        time.sleep(1. / 240.)

                    f_right = p.getBasePositionAndOrientation(right_finger)[0]
                    f_left = p.getBasePositionAndOrientation(left_finger)[0]
                    end_pos_r = np.copy(f_right)
                    end_pos_l = np.copy(f_left)
                    end_pos_l[:2] += (end_pos_l[:2] - np.asarray(p.getBasePositionAndOrientation(sdfId)[0][:2])) * 1.5
                    end_pos_r[:2] += (end_pos_r[:2] - np.asarray(p.getBasePositionAndOrientation(sdfId)[0][:2]) ) * 1.5
                    t_right = 0
                    t_left = 0
                    # the grasping spheres let the object go
                    for r in range(200):
                        if r <= 200:
                            t_right = f_right + 1 / 2 * (1 - np.cos(np.pi * r / 200)) * (end_pos_r - f_right)
                            t_left = f_left + 1 / 2 * (1 - np.cos(np.pi * r / 200)) * (end_pos_l - f_left)

                        p.changeConstraint(right_finger_con, t_right)
                        p.changeConstraint(left_finger_con, t_left)

                        p.stepSimulation()
                        time.sleep(1. / 240.)
                    p.resetBaseVelocity(sdfId, [0, 0, 0], [0, 0, 0])
                    print("this attempt took", grasp_nr, "grasping attempts \n")
                    print("this attempt took an average of", avg_iterations / grasp_nr, "iterations and an average of",
                          avg_time / grasp_nr, "seconds \n")

                    total_avg_time += avg_time / grasp_nr
                    total_avg_iterations += avg_iterations / grasp_nr
                    totalGrasps += grasp_nr
                    successful_runs += 1
                    finished = True
                    time.sleep(1)
                    break
                else:
                    reGrasp = True

                p.stepSimulation()
                time.sleep(1. / 240.)

            p.disconnect()

        print(f"total Grasps: {totalGrasps}")
        print(f"total Iterations: {total_avg_iterations}")
        print(f"total Average Time: {total_avg_time} \n")
        print("--" * 50, "\n")
        print(f"the grasping of the {model} took an average of, {totalGrasps / successful_runs} grasp attempts over "
              f"{successful_runs} successful runs\n")
        print(f"the calculation for the optimal position for the grasping of the {model} took an average of "
              f"{total_avg_iterations / successful_runs} iterations and an average of "
              f"{total_avg_time / successful_runs} seconds, over {successful_runs} successful runs \n")
        print(f"We succeeded {successful_runs} times out of {nr_runs} runs \n")
        print("--" * 50)
        time.sleep(1)
        # this will cause an error if there is no console output to log file set up
        shutil.copyfile(r"../logs/grasp/logs.txt", r"../logs/grasp/" + model + ".txt")
