from controller import Supervisor
import math
import random
import time

TIME_STEP = 32
task_counter=0

def create_cone(x,y,z,name=""):
    children_field = root_node.getField('children')
    cone_vrml = f"""
    DEF CONE_{name} Solid {{
      translation {x} {y} {z}
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 0 0 1 roughness 1 metalness 0 }}
          geometry Cone {{ bottomRadius 0.1 height 0.2 }}
        }}
      ]
      name "{name}"
    }}
    """

    children_field.importMFNodeFromString(-1, cone_vrml)
    
def random_task():
    x = 5*random.random()
    y = 5*random.random()
    z = 0.09
    return (x,y,z)
    
def calc_distance(robot_pos,target_pos): # expecting a 3-tuple

    x_r,y_r,z_r = robot_pos
    x_t,y_t,z_t = target_pos
    
    return math.sqrt((x_r-x_t)**2+(y_r-y_t)**2+(z_r-z_t)**2)



supervisor = Supervisor()  
root_node = supervisor.getRoot()
def_names = ['youbot1', 'youbot2', 'youbot3', 'youbot4']
node_map = {name:supervisor.getFromDef(name) for name in def_names} # contains name:node pairs
translation_fields = {name:node.getField('translation') for name,node in node_map.items()}
tasks={} # global tasks list. 
robot_states={name:'IDLE' for name in def_names}
robot_tasks={}
SPEED=0.05 #m/s
THRESHOLD=0.05

i = 0
while supervisor.step(TIME_STEP) != -1:

    if i%100==0:
        positions = {name:tuple(node_map[name].getPosition()) for name in node_map.keys()}
        # print(positions)       
        
        # simple task allocation: the controller loops through each target and calculates 
        # the distance to each target. Assigns the robot closest to the target to go to target. 
        # every 100 timesteps, give 4 new targets (we assume that the robot is able to finish it's task in 100 timesteps)
        
        for k in range(4):
            task=random_task()
            task_name = f"task_{task_counter}"
            tasks[task_name] = task
            # now create a cone for this task
            create_cone(*task,f'task_{task_counter}') # name it the same as the task. 
            task_counter += 1
        print("tasks generated: ", tasks)
            
# allocation of tasks: allocate only if the robot is idle and there are tasks left to assign         
    idle_robots=[name for name,state in robot_states.items() if state=='IDLE']
    print("CURRENTLY IDLE: ", idle_robots)
    current_positions={name: tuple(node_map[name].getPosition()) for name in idle_robots}
        
    for task_name,task_pos in list(tasks.items()):
        if not idle_robots:
            break
        shortest=math.inf
        closest=None
        for name in idle_robots:
            # where is {name} ?
            pos = current_positions[name]
            r_task = tasks[task_name] # get the task
            # for a task we only want to store the robot closest to the task 
            distance = calc_distance(pos,r_task)
            if distance<shortest :
                shortest = distance
                closest=name
        # if we found the closest for this task.
        if closest:
            robot_states[closest] = task_name # the robot is marked busy
            robot_tasks[closest] = task_pos # the robot's current task is stored. 
            idle_robots.remove(closest)
            del tasks[task_name]
             
    
    # now for execution: 
    for robot in def_names:
        # move only if marked as busy 
        if robot_states[robot]!='IDLE':
            # the coordinates it is supposed to go to: 
            target_position = robot_tasks[robot]   
            current_position = node_map[robot].getPosition()
            task_name = robot_states[robot]
            distance_to_target = calc_distance(current_position, target_position)
            if distance_to_target<THRESHOLD:
                translation_fields[robot].setSFVec3f(list(target_position))
                def_name = f"CONE_{task_name}"
                cone_node = supervisor.getFromDef(def_name)
                if cone_node:
                    cone_node.remove()
                robot_states[robot] = 'IDLE'
                del robot_tasks[robot]
            delta_x = target_position[0]-current_position[0]
            delta_y = target_position[1]-current_position[1]
            v_x = (delta_x/distance_to_target)*SPEED
            v_y = (delta_y/distance_to_target)*SPEED
            new_x = current_position[0]+v_x
            new_y = current_position[1]+v_y
            translation_fields[robot].setSFVec3f([new_x, new_y, current_position[2]])
            
            
            
                        
            


    i += 1