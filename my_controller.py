from controller import Supervisor
import math

def create_cone(x,y,z,name=""):
    children_field = root_node.getField('children')
    cone_vrml = f"""
    Solid {{
      translation {x} {y} {z}
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 0 0 1 roughness 1 metalness 0 }}
          geometry Cone {{ bottomRadius 0.1 height 0.2 }}
        }}
      ]
      name "task_cone_{name}"
    }}
    """

    children_field.importMFNodeFromString(-1, cone_vrml)
    
def calc_distance(robot_pos,target_pos): # expecting a 3-tuple

    x_r,y_r,z_r = robot_pos
    x_t,y_t,z_t = target_pos
    
    return math.sqrt((x_r-x_t)**2+(y_r-y_t)**2+(z_r-z_t)**2)

TIME_STEP = 32

robot = Supervisor()  

root_node = robot.getRoot()
def_names = ['youbot1', 'youbot2', 'youbot3', 'youbot4']
node_map = {name:robot.getFromDef(name) for name in def_names} # contains name:node pairs
translation_fields = {name:node.getField('translation') for name,node in node_map.items()}
print(node_map)
positions = {name:tuple(node_map[name].getPosition()) for name in node_map.keys()}
print(positions)


# simple task allocation: the controller loops through each target and calculates 
# the distance to each target. Assigns the robot closest to the target to go to it. 
tasks = {'task_1':(1.5,-2.0,0.0989),'task_2':(2.5,1.0,0.0989),'task_3':(-1.5,3.0,0.0989), 'task_4':(-2.5,-1.0,0.0989)}
for (x,y,z) in tasks.values():
    create_cone(x,y,z)

task_robot_mapping ={}
# naive approach -  pitfall: if only one of the youbots is the closest to all targets, it allocated to all of them. 
# solution: to track which robot is occupied and which else are free. 


for task in tasks.keys():
    shortest=['',math.inf]
    for name in def_names:
        # where is {name} ?
        pos = positions[name]
        r_task = tasks[task]
        # for a task we only want to store the robot closest to the task 
        distance = calc_distance(pos,r_task)
        if distance<shortest[1]:
            shortest[0]=name
            shortest[1]=distance
    # we found the shortest distance for this task. 
    task_robot_mapping[task] = shortest
         
print(task_robot_mapping)


i = 0
while robot.step(TIME_STEP) != -1:

    if i==0:
        for task_name,assignee_list in task_robot_mapping.items():
            assignee = assignee_list[0]
            new_pos = tasks[task_name] # assign it the position of the task. 
            translation_fields[assignee].setSFVec3f(list(new_pos))

    i += 1