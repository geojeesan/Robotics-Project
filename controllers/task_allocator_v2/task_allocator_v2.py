ENABLE=True
if not ENABLE:
    exit()
from controller import Supervisor
import json
import math
import numpy as np 
import random
from queue import PriorityQueue

TIME_STEP=32
supervisor=Supervisor()
receiver=supervisor.getDevice("receiver")
receiver.enable(TIME_STEP)
emitter=supervisor.getDevice("emitter")
# load precomputed .npy file. This will be used across robots
# want to map trashes and robot positions onto this map. 
occupancy_grid= np.fliplr(np.load("final_map.npy"))

grid_height, grid_width = occupancy_grid.shape
# creating a cell:neighbour map
cell_neighbour_map={}
for i in range(grid_height):
    for j in range(grid_width):
        neighbours={"N":0,"S":0,"E":0,"W":0}
        try:
            if occupancy_grid[i+1][j] == 0:
                neighbours["S"] = 1
        except Exception:
            neighbours["S"] = 1
        try:
            if occupancy_grid[i][j+1] == 0:
                neighbours["E"] = 1
        except Exception:
            neighbours["E"] = 1
        try:
            if i == 0 or occupancy_grid[i-1][j] == 0:
                neighbours["N"] = 1
        except Exception:
            neighbours["N"] = 1
        try:
            if j == 0 or occupancy_grid[i][j-1] == 0:
                neighbours["W"] = 1
        except Exception:
            neighbours["W"] = 1
        cell_neighbour_map[(i,j)] = neighbours
# map robot positions 
MAP_RES = 0.1
MAP_ORIGIN_X= -15
MAP_ORIGIN_Y = -15
NUM_BOTTLES=30
NUM_ROBOTS=4
# robot_list=[f"youbot_{i+1}" for i in in range(NUM_ROBOTS)]
root_node = supervisor.getRoot()
children_field = root_node.getField('children')


# def create_robots(NUM_ROBOTS):
    # with open "robot_vrml"

def world2grid(world_x,world_y):
    rel_x = world_x - MAP_ORIGIN_X
    rel_y = world_y - MAP_ORIGIN_Y
    gx = int(rel_x / MAP_RES)
    gy = int(rel_y / MAP_RES)
    # gx -> col, gy -> row
    if 0 <= gx < grid_width and 0 <= gy < grid_height: # check if it is in bounds or not
        return gy, gx # row,col
    else:
        # outside known map
        return None
        
def grid2world(row,col):
    # row is y, col is x
    world_x = MAP_ORIGIN_X + (col + 0.5) * MAP_RES
    world_y = MAP_ORIGIN_Y + (row + 0.5) * MAP_RES
    return world_x, world_y

assignable_coords = set([grid2world(row,col) for row in range(grid_height) for col in range(grid_width) if occupancy_grid[row][col]==1])
num_to_create = min(NUM_BOTTLES, len(assignable_coords))
chosen = set(random.sample(assignable_coords, num_to_create))
available_positions= assignable_coords-chosen

def randomise_robot_orientations():
    youbot_list=["youbot_1","youbot_2","youbot_3","youbot_4"]
    for robot in youbot_list:
        robot_node=supervisor.getFromDef(robot)
        rotation_field=robot_node.getField('rotation')
        random_rotation=np.random.uniform(-math.pi,math.pi)
        rot=[0,0,1,random_rotation]
        rotation_field.setSFRotation(rot) 
        trans_field=robot_node.getField("translation")
        random_position = random.choice(list(available_positions))
        w_x,w_y= random_position[0],random_position[1]
        trans_field.setSFVec3f([w_x,w_y,0.0986])
 

def create_bottles(occupancy_grid):
    for i, (b_x, b_y) in enumerate(chosen):
        bottle_vrml = f"""
        WaterBottle {{
          translation  {b_x} {b_y} 0
          rotation 0.8435777383914596 0.49366584760060905 -0.21135427651959723 1.5830930893672932e-16
          name "water_bottle_{i}"
        }}
        """
        children_field.importMFNodeFromString(-1, bottle_vrml)


def create_cone(x, y, z, name="",scale=1.0):
    root_node = supervisor.getRoot()
    children_field = root_node.getField('children')
    robot_name=""
    size=0.6*scale
    for i in ["youbot_1","youbot_2","youbot_3","youbot_4"]:
        if i in name:
            robot_name=i
            break
    colors = {"youbot_1": "1 0 0", "youbot_2": "0 1 0", "youbot_3": "0 0 1", "youbot_4": "1 1 0"}
    c_str = colors.get(robot_name, "0.5 0.5 0.5") # grey cone for any other name. 
    cone_vrml = f"""
    DEF {name} Solid {{
      translation {x} {y} {z}
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor {c_str} roughness 1 metalness 0 transparency 0.5 }}
          geometry Cone {{ bottomRadius 0.05 height {size} }}
        }}
      ]
      name "{name}"
    }}
    """
    children_field.importMFNodeFromString(-1, cone_vrml)

                       
def h(cell_1,cell_2): # cell_1 and cell_2 are tuples
    return abs(cell_1[0]-cell_2[0])+abs(cell_1[1]-cell_2[1])

A_STAR_DEBUG = False

def a_star(start_cell,goal_cell,occupancy_grid):
    """
    Return ordered list of grid cells from start_cell -> goal_cell (inclusive),
    or None if no path exists.
    """
    grid_height, grid_width = occupancy_grid.shape

    if start_cell is None or goal_cell is None:
        return None

    if start_cell == goal_cell:
        return [start_cell]

    sr, sc = start_cell
    gr, gc = goal_cell
    if not (0 <= sr < grid_height and 0 <= sc < grid_width):
        return None
    if not (0 <= gr < grid_height and 0 <= gc < grid_width):
        return None

    if occupancy_grid[sr, sc] == 0 or occupancy_grid[gr, gc] == 0:
        return None

    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    g_score = {start_cell: 0}
    f_score = {start_cell: heuristic(start_cell, goal_cell)}
    open_q = PriorityQueue()
    open_q.put((f_score[start_cell], heuristic(start_cell, goal_cell), start_cell))
    came_from = {}
    visited = set()

    while not open_q.empty():
        _, _, curr_cell = open_q.get()
        if curr_cell in visited:
            continue
        visited.add(curr_cell)
        if curr_cell == goal_cell:
            # reconstruct ordered path
            path = []
            node = goal_cell
            while node != start_cell:
                path.append(node)
                node = came_from.get(node)
                if node is None:
                    return None
            path.append(start_cell)
            path.reverse()
            return path

        # neighbors (cell_neighbour_map uses 0 for free, 1 for blocked/out-of-bounds)
        for d in "NSEW":
            if cell_neighbour_map[curr_cell].get(d, 1) != 0:
                continue
            if d=="N":
                child_cell = (curr_cell[0]-1,curr_cell[1])
            elif d=="S":
                child_cell = (curr_cell[0]+1,curr_cell[1])
            elif d=="E":
                child_cell = (curr_cell[0],curr_cell[1]+1)             
            elif d=="W":
                child_cell = (curr_cell[0],curr_cell[1]-1)

            nr, nc = child_cell
            if not (0 <= nr < grid_height and 0 <= nc < grid_width):
                continue
            if occupancy_grid[nr, nc] == 0:
                continue

            tentative_g = g_score.get(curr_cell, float('inf')) + 1
            if tentative_g < g_score.get(child_cell, float('inf')):
                came_from[child_cell] = curr_cell
                g_score[child_cell] = tentative_g
                f_score[child_cell] = tentative_g + heuristic(child_cell, goal_cell)
                open_q.put((f_score[child_cell], heuristic(child_cell, goal_cell), child_cell))

    return None


def grid_path_to_world(path_cells):
    return [grid2world(r, c) for (r, c) in path_cells]

def get_path_cost(start_pos_w,goal_pos_w):
    """ 
    This function returns (path_world_list, cost_int) or (None, inf) if no path
    """
    start_cell = world2grid(start_pos_w[0],start_pos_w[1])
    goal_cell = world2grid(goal_pos_w[0],goal_pos_w[1])

    if start_cell is None or goal_cell is None:
        return None, float('inf')

    path_cells = a_star(start_cell, goal_cell, occupancy_grid)
    if path_cells is None:
        return None, float('inf')

    path_world = grid_path_to_world(path_cells)
    cost = max(0, len(path_cells)-1)
    return path_world, cost

robots_state={} # robot:{x:"",y:"",state:"INIT/EXECUTING"}
global_trash_map=[] # list of {trashes:{x:"",y:""}}


def merge_trash_data(new_trash_list):
    global global_trash_map
    for new_trash in new_trash_list:
        grid_cell = world2grid(new_trash['x'],new_trash['y'])
        if grid_cell is None:
            print(f"world2grid returned None for world coordinates x={new_trash['x']:.2f},y={new_trash['y']:.2f} ")
            continue
        row,col = grid_cell
        final_row,final_col=row,col
        # occupancy_grid uses 1=free, 0=obstacle -> treat 0 as invalid location
        if occupancy_grid[row][col] == 0: # obstacle is actually misidentified trash
            print(f"({row},{col}) is an invalid location, searching its neighbours")
            candidates=[]
            neighbours = cell_neighbour_map.get((row,col),{})
            print(f"Neighbours of ({row},{col}) are {neighbours}")
            if row - 1 >= 0 and neighbours.get("N", 1) == 0 and occupancy_grid[row - 1][col] == 1:
                candidates.append((row - 1, col))
            if row + 1 < grid_height and neighbours.get("S", 1) == 0 and occupancy_grid[row + 1][col] == 1:
                candidates.append((row + 1, col))
            if col + 1 < grid_width and neighbours.get("E", 1) == 0 and occupancy_grid[row][col + 1] == 1:
                candidates.append((row, col + 1))
            if col - 1 >= 0 and neighbours.get("W", 1) == 0 and occupancy_grid[row][col - 1] == 1:
                candidates.append((row, col - 1))
            
            # pick the closest candidate
            if not candidates:
                # do a radius 2 sweep if radius 1 fails. 
                print("doing a radius 2 sweep")
                for dr in (-2, -1, 0, 1, 2):
                    for dc in (-2, -1, 0, 1, 2):
                        r2, c2 = row + dr, col + dc
                        if r2 < 0 or r2 >= grid_height or c2 < 0 or c2 >= grid_width: 
                            continue
                        if occupancy_grid[r2][c2] == 1:
                            candidates.append((r2, c2))
                    if candidates:
                        break                
            if candidates:                
                candidates.sort(key=lambda rc:h(rc,(row,col)))
                final_row,final_col=candidates[0]
                print(f"Found neighbour ({final_row},{final_col}) instead. ")
            else:
                print("no neighbours found")
                continue
            
        final_x,final_y = grid2world(final_row,final_col) # this is now in world coordinates    
        is_dup = False
        for existing_trash in global_trash_map:
            d=math.sqrt((existing_trash['x']-final_x)**2+(existing_trash['y']-final_y)**2)
            if d<0.6: # duplicate threshold
                is_dup=True
                break
        if not is_dup: # its a new trash at a new loc
            t_id =  f"{final_x:.2f}_{final_y:.2f}"
            global_trash_map.append({
                'x': final_x,
                'y': final_y,
                'id': t_id,
                'assigned_to':None
            }) 
            print(f"New trash added at ({final_x:.2f}, {final_y:.2f})")
# supervisor main 

print("listening for data from robots...")
# put random bottles

create_bottles(occupancy_grid)
randomise_robot_orientations()
create_cone(-2.3504,6.65,-0.20,"test")
while supervisor.step(TIME_STEP) != -1:
    while receiver.getQueueLength() > 0:
        packet=receiver.getString()
        data=json.loads(packet) 
        receiver.nextPacket()     
        robot = data['robot']
        print(f"received data from {robot}")
        
        robots_state[robot] = {
            'x': data['pose']['x'],
            'y':data['pose']['y'],
            'theta': data['pose']['theta'],
            'status': data['status']
        }
        if len(data['trash_world_coords'])>0:
            print(f"number of trashes found by {robot} = {len(data['trash_world_coords'])}")
            merge_trash_data(data['trash_world_coords'])
     

    available_robots=[name for name, state in robots_state.items() if state['status'] == 'WAITING']
    unassigned_trash = any(t['assigned_to'] is None for t in global_trash_map)
    
    if available_robots and unassigned_trash:
        assignments=[]
        cost_cache = {}
        for robot in available_robots:
            r_pos = (robots_state[robot]['x'],robots_state[robot]['y'])
            for t_idx,trash in enumerate(global_trash_map):
                if trash['assigned_to'] is None:
                    t_pos = (trash['x'],trash['y'])
                    key = (robot, t_idx)
                    if key in cost_cache:
                        path_world, cost = cost_cache[key]
                    else:
                        path_world, cost = get_path_cost(r_pos,t_pos)
                        cost_cache[key] = (path_world, cost)
                    if math.isinf(cost):
                        print(f"cost is infinite to {trash['x']},{trash['y']} ")
                        continue
                    assignments.append({
                        'cost':cost,
                        'robot': robot,
                        'trash_idx':t_idx,
                        'path': path_world
                    })
                
                
        assignments.sort(key=lambda x: x['cost']) # we choose the robot closest
        assigned_robots=set() # no duplicates
        assigned_trash_indices=set()
        commands={}
        
        for option in assignments:
            r=option['robot']
            t_idx=option['trash_idx']
            if r not in assigned_robots and t_idx not in assigned_trash_indices:
                assigned_robots.add(r)
                assigned_trash_indices.add(t_idx)
                global_trash_map[t_idx]['assigned_to'] = r
                robots_state[r]['status'] = "BUSY"
                target_x = global_trash_map[t_idx]['x']
                target_y = global_trash_map[t_idx]['y']
                commands[r] = {
                    "goto_x":target_x,
                    "goto_y":target_y,
                    "trash_id": global_trash_map[t_idx]['id'],
                    "path": option['path']
                }
                
                print(f"Assigning {r} -> Trash {t_idx} (Dist: {option['cost']:.2f})")
                create_cone(target_x,target_y,0,f"{r}")
                for i,path in enumerate(option['path']):
                    create_cone(path[0],path[1],0,f"path_cone_{r}_{i}")
                    
                    
        if commands:
            msg=json.dumps(commands)
            print(msg)
            emitter.send(msg.encode('utf-8'))
            print("sent tasks")

