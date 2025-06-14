import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import random
import itertools
import json

GRID_SIZE = 10
spatial_hash = set()
solution_set = []
bins = [0,0,0,0,0,0,0,0]
bin_limit = 125
depth_limit = 5
iteration_count = 0

PENTOMINOES = {
    'I': [(0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0)],
    'V': [(0,0,0),(1,0,0),(2,0,0),(2,1,0),(2,2,0)],
    'F': [(0,0,0),(1,0,0),(1,1,0),(1,2,0),(2,1,0)],
    'L': [(0,0,0),(1,0,0),(1,1,0),(1,2,0),(1,3,0)],
    'N': [(0,0,0),(0,1,0),(1,1,0),(1,2,0),(1,3,0)],
    'P': [(0,0,0),(0,1,0),(1,0,0),(1,1,0),(1,2,0)],
    'T': [(0,0,0),(1,0,0),(2,0,0),(1,1,0),(1,2,0)],
    'Z': [(0,0,0),(1,0,0),(1,1,0),(1,2,0),(2,2,0)],
    'U': [(0,0,0),(0,1,0),(1,0,0),(2,0,0),(2,1,0)],
    'W': [(0,0,0),(1,0,0),(1,1,0),(2,1,0),(2,2,0)],
    'Y': [(0,0,0),(0,1,0),(0,2,0),(1,2,0),(0,3,0)],
    'X': [(0,1,0),(1,0,0),(1,1,0),(2,1,0),(1,2,0)],
}

rotation_matrices = np.array([
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
    [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
    [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
    [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
    [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
    [[-1, 0, 0], [0, 0, -1], [0, -1, 0]],
    [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
    [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
    [[0, 0, -1], [0, -1, 0], [-1, 0, 0]],
    [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
    [[0, 0, 1], [0, -1, 0], [1, 0, 0]],
    [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
    [[0, -1, 0], [0, 0, 1], [1, 0, 0]],
    [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
    [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
    [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
])
reflections = [np.diag(r) for r in itertools.product([-1,1], repeat=3)]

def normalize(coords):
    x_coord = []
    y_coord = []
    z_coord = []
    for point in coords:
        x_coord.append(point[0])
        y_coord.append(point[1])
        z_coord.append(point[2])
    min_x,min_y,min_z = min(x_coord),min(y_coord),min(z_coord)
    return tuple(sorted((x-min_x,y-min_y,z-min_z) for (x,y,z) in coords))

def rotation(coords):
    coords_np = np.array(coords)
    rots = set()
    for mat in rotation_matrices:
        rotated = np.dot(coords_np,mat.T)
        rotated = np.round(rotated).astype(int)
        rotated_norm = normalize(rotated)
        rots.add(rotated_norm)
        for r in reflections:
            reflected = np.dot(rotated_norm,r.T)
            reflected = np.round(reflected).astype(int)
            reflected_norm = normalize(reflected)
            rots.add(reflected_norm)
    unique_rots = list(rots)
    return unique_rots
    
def valid_placement(grid,coords,space_constraint,placed):
    freestanding = False
    for (x,y,z) in coords:
        if x >= GRID_SIZE or y >= GRID_SIZE or z >= GRID_SIZE or x < 0 or y < 0 or z < 0:
            return False
    for (x,y,z) in coords:
        if grid[x,y,z]:
            return False
    for (x,y,z) in coords:
        if z==0:
            freestanding = True
        else:
            if grid[x,y,z-1]:
                freestanding = True
    if freestanding:
        score = 0
        if placed == 0:
            return True
        for (x,y,z) in coords:
            for dx,dy,dz in [(-1,0,0),(1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                if 0 <= x+dx < GRID_SIZE and 0 <= y+dy < GRID_SIZE and 0 <= z+dz < GRID_SIZE:
                    if grid[x+dx,y+dy,z+dz]:
                        score += 1
        if score < space_constraint:
            return False
    else:
        return False
    return True

def canonicalize(coords):
    rots = rotation(coords)
    return min(rots)

def backtrack(pentomino_configs):
    global iteration_count
    queue = deque([{
        'placed': [],
        'used': set(),
        'occupied': [],
        'grid': np.full((GRID_SIZE,GRID_SIZE,GRID_SIZE), False),
    }])
    while queue:
        iteration_count += 1
        state = queue.popleft();
        depth = len(state['used'])
        
        if len(solution_set) >= bin_limit*len(bins):
            break
                
        if len(state['used']) == len(PENTOMINOES):
            continue

        if depth >= 5:
            if bins[depth-5] < bin_limit:
                bins[depth-5] += 1
                solution_set.append(state['placed'])
                if len(solution_set) % 100 == 0:
                    print(bins,"Total solutions:",len(solution_set))
        
        children = []
        for name,shape in pentomino_configs.items():
            if name in state['used']:
                continue
            for config in shape:
                for dx in range(GRID_SIZE):
                    for dy in range(GRID_SIZE):
                        for dz in range(GRID_SIZE):
                            coords = [(x+dx,y+dy,z+dz) for (x,y,z) in config]
                            if valid_placement(state['grid'],coords,min(4,max(2,(len(list(state['used']))+1)*2//3)),len(list(state['used']))):
                                updated_coords = list(state['occupied'])
                                updated_coords.append(coords)
                                updated_grid = np.copy(state['grid'])
                                for (x,y,z) in coords:
                                    updated_grid[x,y,z] = True
                                all_coords = list(itertools.chain.from_iterable(updated_coords))
                                min_state = canonicalize(all_coords)
                                if min_state in spatial_hash:
                                    continue
                                spatial_hash.add(min_state)
                                children.append({
                                    'placed': state['placed'] + [(name, coords)],
                                    'used': state['used'] | {name},
                                    'grid': updated_grid,
                                    'occupied': updated_coords,
                                })
        random.shuffle(children)
        for child in children[:depth_limit]:
            queue.append(child)
        if iteration_count % 50 == 0:
            temp_list = list(queue)
            random.shuffle(temp_list)
            queue = deque(temp_list)

def display_solution(coords,ax):
    filled = np.zeros((GRID_SIZE,GRID_SIZE,GRID_SIZE),dtype=bool)
    for coord in coords:
        for x,y,z in coord:
            filled[z,y,x] = True
    ax.voxels(filled)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

class NumpyEncoder(json.JSONEncoder):
    def default(self,obj):
        if isinstance(obj, (np.integer)):
            return int(obj)
        elif isinstance(obj, (np.floating)):
            return float(obj)
        elif isinstance(obj, (np.ndarray)):
            return obj.tolist()
        return super().default(obj)

def json_helper(solution):
    return [
        {
            'piece': piece,
            'coords': [list(coord) for coord in coords]
        }
        for piece,coords in solution
    ]
    

if __name__ == "__main__":
    pentomino_configs = {}
    for name,shape in PENTOMINOES.items():
        rots = rotation(shape)
        pentomino_configs[name] = rots
    backtrack(pentomino_configs)
    fig = plt.figure(figsize=(100,100))
    print(len(solution_set))
    
    solutions_list = [json_helper(soln) for soln in solution_set]
    with open('solutions.json','w') as f:
        json_data = json.dumps(solutions_list,cls=NumpyEncoder)
        f.write(json_data)
    
    for i in range(10):
        piece,coords = zip(*solution_set[i])
        ax = fig.add_subplot(2,5,i+1,projection='3d')
        display_solution(coords,ax=ax)
    plt.show()
