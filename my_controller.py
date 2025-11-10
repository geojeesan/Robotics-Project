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



robot = Supervisor()  
root_node = robot.getRoot()
def_names = ['youbot1', 'youbot2', 'youbot3', 'youbot4']
node_map = {name:robot.getFromDef(name) for name in def_names} # contains name:node pairs
translation_fields = {name:node.getField('translation') for name,node in node_map.items()}
tasks={} # global tasks list. 

i = 0
while robot.step(TIME_STEP) != -1:

    if i%100==0:
        positions = {name:tuple(node_map[name].getPosition()) for name in node_map.keys()}
        # print(positions)       
        
        # simple task allocation: the controller loops through each target and calculates 
        # the distance to each target. Assigns the robot closest to the target to go to target. 
        # every 100 timesteps, give 4 new targets (we assume that the robot is able to finish it's task in 100 timesteps)
        
        for k in range(8):
            task=random_task()
            task_name = f"task_{task_counter}"
            tasks[task_name] = task
            # now create a cone for this task
            create_cone(*task,f'task_{task_counter}') # name it the same as the task. 
            task_counter += 1
        print("tasks assigned: ", tasks)
            
         
    if tasks: # while tasks is not empty, then create a mapping and do them.   
        #  (✿◕‿◕✿)
        task_robot_mapping ={}
        
        available_robots = def_names.copy()
        
        for task_name,task_pos in tasks.items():
            shortest=math.inf
            closest=None
            for name in available_robots:
                # where is {name} ?
                pos = positions[name]
                r_task = tasks[task_name] # get the task
                # for a task we only want to store the robot closest to the task 
                distance = calc_distance(pos,r_task)
                if distance<shortest and name in available_robots:
                    shortest = distance
                    closest=name
            # if we found the closest for this task.
            if closest:
                task_robot_mapping[task_name] = closest
                available_robots.remove(closest) 
                 
        print("TODO: ", task_robot_mapping) # this is the allocation. 
        
        for task_name,assignee in task_robot_mapping.items():
            new_pos = tasks[task_name] # assign it the position of the task. 
            translation_fields[assignee].setSFVec3f(list(new_pos)) # teleport!! ( ͡° ͜ʖ ͡°)
            print(f"{task_name} completed by {assignee}")
            # remove cone associated with the task name.
            def_name = f"CONE_{task_name}"
            cone_node = robot.getFromDef(def_name)
            cone_node.remove()
            # remove task from queue
            del tasks[task_name]
            print(f"Remaining Tasks {tasks}")

    i += 1