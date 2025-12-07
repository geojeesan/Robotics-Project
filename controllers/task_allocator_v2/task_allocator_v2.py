ENABLE=True
if not ENABLE:
    exit()
from controller import Supervisor
import json
import math
import numpy as np 
from queue import PriorityQueue

TIME_STEP=32
supervisor=Supervisor()
receiver=supervisor.getDevice("receiver")
receiver.enable(TIME_STEP)
emitter=supervisor.getDevice("emitter")
# load precomputed .npy file. This will be used across robots
# want to map trashes and robot positions onto this map. 
occupancy_grid= np.load("final_map.npy")
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

def create_cone(x, y, z, name=""):
    root_node = supervisor.getRoot()
    children_field = root_node.getField('children')
    robot_name=name
    colors = {"youbot_1": "1 0 0", "youbot_2": "0 1 0", "youbot_3": "0 0 1", "youbot_4": "1 1 0"}
    c_str = colors.get(robot_name, "0.5 0.5 0.5")
    cone_vrml = f"""
    DEF {name} Solid {{
      translation {x} {y} {z}
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor {c_str} roughness 1 metalness 0 transparency 0.5 }}
          geometry Cone {{ bottomRadius 0.05 height 0.6 }}
        }}
      ]
      name "{name}"
    }}
    """
    children_field.importMFNodeFromString(-1, cone_vrml)

def world2grid(world_x,world_y):
    rel_x = world_x - MAP_ORIGIN_X
    rel_y = world_y - MAP_ORIGIN_Y
    gx = int(rel_x / MAP_RES)
    gy = int(rel_y / MAP_RES)
    # print("gx: ",gx)
    # print("gy: ",gy)
    if 0 <= gx < grid_width and 0 <= gy < grid_height: # check if it is in bounds or not
        return gy, gx # row,col
    else:
        # Robot is outside the known map
        return None
        
def h(cell_1,cell_2): # cell_1 and cell_2 are tuples
    return abs(cell_1[0]-cell_2[0])+abs(cell_1[1]-cell_2[1])
    

A_STAR_DEBUG = False

def a_star(start_cell,goal_cell,occupancy_grid):
    grid_height, grid_width = occupancy_grid.shape

    if start_cell is None or goal_cell is None:
        return None

    if start_cell == goal_cell:
        return {}

    sr, sc = start_cell
    gr, gc = goal_cell
    if not (0 <= sr < grid_height and 0 <= sc < grid_width):
        return None
    if not (0 <= gr < grid_height and 0 <= gc < grid_width):
        return None

    # now occupancy_grid uses 1=free, 0=obstacle -> blocked if == 0
    if occupancy_grid[sr, sc] == 0 or occupancy_grid[gr, gc] == 0:
        return None

    g_score={}
    for i in range(grid_height):
        for j in range(grid_width):
            g_score[(i,j)]=float('inf')
    f_score={k:float('inf') for k in g_score}
    g_score[start_cell]=0
    f_score[start_cell]=h(start_cell,goal_cell)
    
    open=PriorityQueue()
    open.put((f_score[start_cell],h(start_cell,goal_cell),start_cell))
    came_from={}
    visited=set()
    
    while not open.empty():
        curr_cell = open.get()[2]
        if curr_cell in visited:
            continue
        visited.add(curr_cell)
        if curr_cell==goal_cell:
            # reconstruct
            fwd_path={}
            node = goal_cell
            while node!= start_cell:
                parent = came_from.get(node)
                if parent is None:
                    return None
                fwd_path[parent]= node
                node = parent
            return fwd_path
        for d in "NSEW":
            if cell_neighbour_map[curr_cell][d]==0:
                if d=="N":
                    child_cell = (curr_cell[0]-1,curr_cell[1])
                if d=="S":
                    child_cell = (curr_cell[0]+1,curr_cell[1])
                if d=="E":
                    child_cell = (curr_cell[0],curr_cell[1]+1)             
                if d=="W":
                    child_cell = (curr_cell[0],curr_cell[1]-1)

                nr, nc = child_cell
                if not (0 <= nr < grid_height and 0 <= nc < grid_width):
                    continue
                # blocked if occupancy_grid==0
                if occupancy_grid[nr, nc] == 0:
                    continue

                temp_g_score=g_score[curr_cell]+1
                temp_f_score=temp_g_score+h(child_cell,goal_cell)
                
                if temp_f_score<f_score.get(child_cell, float('inf')):
                    g_score[child_cell] = temp_g_score
                    f_score[child_cell] = temp_f_score
                    open.put((temp_f_score,h(child_cell,goal_cell),child_cell))
                    came_from[child_cell] = curr_cell
    return None


def grid2world(row,col):
    # row is y, col is x
    world_x = MAP_ORIGIN_X + (col + 0.5) * MAP_RES
    world_y = MAP_ORIGIN_Y + (row + 0.5) * MAP_RES
    return world_x, world_y
                       

    
def get_path_cost(start_pos_w,goal_pos_w):
    """ 
    This function returns the cost of going from 
    start_pos to goal_pos given the map    
    """
    start_pos= world2grid(start_pos_w[0],start_pos_w[1])
    
    
    goal_pos= world2grid(goal_pos_w[0],goal_pos_w[1])
    
    print("does grid2world (world2grid) == world ? ")
    print(f"start world coords: {start_pos_w}")
    print(f"world2grid {start_pos_w} = {world2grid(*start_pos_w)}")
    print(f"grid2world {start_pos} = {grid2world(*start_pos)}")
    p = a_star(start_pos,goal_pos,occupancy_grid)
    if p is None:
        return float('inf')
    else:
        p= [grid2world(*coord) for coord in p]
    return p,len(p) # just g()    

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
            # lets look at its neighbors and assign it to the closest available free neighbor \
            # surely one of them will be free (right)??
            candidates=[]
            neighbours = cell_neighbour_map.get((row,col),{}) # this will be of the form (row,col):{"N":[0/1],"S":[0/1]...}
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
                        # pick free cells (occupancy_grid==1)
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
            if d<0.8: # same as the one we used for the robots
                is_dup=True
                break
        if not is_dup: # its a new trash at a new loc
            # if multiple robots detect the same trash, we want to untangle it.
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
while supervisor.step(TIME_STEP) != -1:
    while receiver.getQueueLength() > 0:
        packet=receiver.getString()
        data=json.loads(packet) 
        # print(data)   
        receiver.nextPacket()     
        robot = data['robot']
        print(f"received data from {robot}")
        
        # current_status= robots_state[robot]['status']
            
        robots_state[robot] = {
            'x': data['pose']['x'],
            'y':data['pose']['y'],
            'theta': data['pose']['theta'],
            'status': data['status'] # this will be WAITING because the robots only communicate with the server for assignments and they only ask for assignments when they are in WAITING.
        }
        if len(data['trash_world_coords'])>0:
            print(f"number of trashes found by {robot} = {len(data['trash_world_coords'])}")
            merge_trash_data(data['trash_world_coords'])
     


            
    available_robots=[name for name, state in robots_state.items() if state['status'] == 'WAITING']
    #
    unassigned_trash = any(t['assigned_to'] is None for t in global_trash_map)
    # 
    
    if available_robots and unassigned_trash:
        # find the distance of each robot to trash: 
        print(f"Available robots: {available_robots}")
        print(f"Unassigned trash: {unassigned_trash}")
        assignments=[]
        cost_cache = {}
        for robot in available_robots:
            r_pos = (robots_state[robot]['x'],robots_state[robot]['y'])
            for t_idx,trash in enumerate(global_trash_map):
                if trash['assigned_to'] is None:
                    t_pos = (trash['x'],trash['y'])
                    key = (robot, t_idx)
                    if key in cost_cache:
                        cost = cost_cache[key]
                    else:
                        p,cost = get_path_cost(r_pos,t_pos)
                        cost_cache[key] = cost
                    if math.isinf(cost):
                        continue
                    assignments.append({
                        'cost':cost,
                        'robot': robot,
                        'trash_idx':t_idx
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
                    "path": p
                }
                
                print(f"Assigning {r} -> Trash {t_idx} (Dist: {option['cost']:.2f})")
                create_cone(target_x,target_y,0,f"{r}")

        if commands:
            msg=json.dumps(commands)
            print(msg)
            emitter.send(msg.encode('utf-8'))
                
