#!/usr/bin/env python3
# -*- coding: utf-8 -*-#

# Caspians project 287776
# pandas for data manipulation
import pandas as pd # took me 2 days to discover this 
# networkx for graph manipulation and dijkstra algorithm
import networkx as nx
# matplotlib for graph visualization alongisde network x
import matplotlib.pyplot as plt
# i forgot why i used random
import random
import time 

# implement A* by next week
# peformance test 
# report 

class stopWatch():
    def __init__(self, name):
        self.elapsed = None
        self.time = None
        self.algorithm = name
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = float(time.time())  # timer for performance testing

    def stop(self):
        self.end_time = float(time.time())
        self.duration()

    def duration(self):
        self.elapsed = self.end_time - self.start_time
        print("Algorithm: {}, time elapsed: {} seconds".format(self.algorithm, self.elapsed))
        return self.elapsed
        

# roughly 30% assisted by AI
def time_to_seconds(time_str):
    # didnt use date time as it was being broken 
    h, m, s = map(int, time_str.split(':'))
    # split the time into hours, minutes and second then return in seconds 
    return h * 3600 + m * 60 + s


def calculate_travel_time(dep, arr):
    # if its midnight its the next day  (sanatisation)
    if arr == "24:00:00": 
        arr = "00:00:00"
    d_sec, a_sec = time_to_seconds(dep), time_to_seconds(arr)
    if a_sec < d_sec:
        a_sec += 86400  # add 24 hours in seconds to arrival time if it is less than departure time
    return a_sec - d_sec

def load_data(fp):
    # load our csv file using pandas
    df = pd.read_csv(fp)
    # initialise our networkx graph
    # its stored in network x not just viusalised, the visualisation occurs in matplotlib
    g = nx.DiGraph()
    for _, r in df.iterrows():
        g.add_edge(r['start_stop'], r['end_stop'], weight=calculate_travel_time(r['departure_time'], r['arrival_time']), line=r['line'])
    return g

def dijkstra(graph, start):
    # distance and previous node 
    dist, prev = {n: float('inf') for n in graph.nodes}, {n: None for n in graph.nodes}
    # distance of the start node is always 0
    dist[start] = 0
    # create our unvisited list of nodes as all the current nodes 
    unvisited = list(graph.nodes)
    # while we have unvisited nodes 
    while unvisited:
        # current = the node in our unvisited with the lowest distance 
        curr = min(unvisited, key=lambda n: dist[n])
        # visit it by removing it from the list 
        unvisited.remove(curr)
        # for the neighbours in the currently visited node 
        for nb in graph.neighbors(curr):
            # get the distance of the distance to the current node + the distance to the neighbour node 
            alt = dist[curr] + graph[curr][nb]['weight']
            # find the lowest distance to the next node from this visited one 
            if alt < dist[nb]:
                dist[nb], prev[nb] = alt, curr
    # honestly i forgot what i did with prev and distance i named them badly
    # but this is the distance from the start node to all other nodes according to copliot that wanted to try comment mor me
    return dist, prev

# builds the path more than gets a stored value 
def get_path(prev, start, end):
    # if theres no route from the last node to the start node return none
    if end not in prev or prev[end] is None:
        print("No route found.")
        return None
    path, cur = [], end
    while cur and cur != start:
        # insert the node to the path from the last to first inserting at the start to reverse the order
        path.insert(0, cur)
        cur = prev.get(cur)
    if cur: 
        # insert the start node if it is the end at the start
        path.insert(0, start)
        # return the path
    return path

# visual graph all done by AI assistance (i dont know how to visualise data
def show_graph(graph, path):
    if not path:
        print("No path to visualize.")
        return
    subgraph = graph.subgraph(path)
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(subgraph, seed=random.randint(0, 100))
    nx.draw(subgraph, pos, with_labels=True, node_color="lightblue", edge_color="red", node_size=700, font_size=10)
    nx.draw_networkx_edges(subgraph, pos, edgelist=list(zip(path, path[1:])), edge_color="red", width=2.5)
    nx.draw_networkx_nodes(subgraph, pos, nodelist=path, node_color="yellow", node_size=700)
    plt.show()

def reconstruct_path(cameFrom, current):
    total_path = [current]
    while current in cameFrom: # check this line
        current = cameFrom[current]
        total_path.append(current)
    return total_path  

def heuristic(graph, node, end):
    # a* heuristic function for A* algorithm
    # Using a simple heuristic: the straight-line distance between nodes
    # Assuming the graph nodes have coordinates stored as attributes 'x' and 'y'
    def euclidean_distance(node1, node2):
        x1, y1 = graph.nodes[node1].get('x', 0), graph.nodes[node1].get('y', 0)
        x2, y2 = graph.nodes[node2].get('x', 0), graph.nodes[node2].get('y', 0)
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    return euclidean_distance(node, end)

def heuristicAlternative(graph, node, end):
    # A simpler heuristic function for A* algorithm
    # Using the degree of the node as a heuristic
    return abs(graph.degree(node) - graph.degree(end))

def a_star(graph, start, end):
    # *a star algorithm implmenetation
    # based on the wikipedia A* code
    open_set = {start}
    cameFrom = {}
    gScore = {node: float('inf') for node in graph.nodes} # infinity for all unknowns  
    gScore[start] = 0
    fScore = {node: float('inf') for node in graph.nodes} # infinity for all unknowns  
    fScore[start] = heuristic(graph, start, end)
    while open_set:
        current = min(open_set, key=lambda n: fScore[n])
        if current == end:
            return reconstruct_path(cameFrom, current)
        open_set.remove(current)
        for neighbor in graph.neighbors(current):
            tentative_gScore = gScore[current] + graph[current][neighbor]['weight']
            if tentative_gScore < gScore[neighbor]:
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = gScore[neighbor] + heuristic(graph, neighbor, end)
                open_set.add(neighbor)
    return None


def tabu_search(graph, start, end, max_iterations=1000, tabu_size=2):
    current = start
    best_path = [current]
    best_cost = float('inf')
    tabu_list = []
    for _ in range(max_iterations):
        neighbors = list(graph.neighbors(current))
        neighbors = [n for n in neighbors if n not in tabu_list]
        if not neighbors:
            break
        next_node = min(neighbors, key=lambda n: graph[current][n]['weight'])
        best_path.append(next_node)
        tabu_list.append(current)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)
        current = next_node
        if current == end:
            best_cost = sum(graph[best_path[i]][best_path[i + 1]]['weight'] for i in range(len(best_path) - 1))
            break
    return best_path if current == end else None, best_cost

def tabu_search_variable_t(graph, start, end, max_iterations=1000):
    current = start
    best_path = [current]
    best_cost = float('inf')
    tabu_list = []
    for _ in range(max_iterations):
        neighbors = list(graph.neighbors(current))
        neighbors = [n for n in neighbors if n not in tabu_list]
        if not neighbors:
            break
        next_node = min(neighbors, key=lambda n: graph[current][n]['weight'])
        best_path.append(next_node)
        tabu_list.append(current)
        if len(tabu_list) > len(best_path) // 2:
            tabu_list.pop(0)
        current = next_node
        if current == end:
            best_cost = sum(graph[best_path[i]][best_path[i + 1]]['weight'] for i in range(len(best_path) - 1))
            break
    return best_path if current == end else None, best_cost

def tabu_search_with_aspiration(graph, start, end, max_iterations=100, tabu_size=1):
    current = start
    best_path = [current]
    best_cost = float('inf')
    tabu_list = []
    for _ in range(max_iterations):
        neighbors = list(graph.neighbors(current))
        next_node = None
        for n in neighbors:
            if n not in tabu_list or graph[current][n]['weight'] < best_cost:
                next_node = n
                break
        if not next_node:
            break
        best_path.append(next_node)
        tabu_list.append(current)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)
        current = next_node
        if current == end:
            best_cost = sum(graph[best_path[i]][best_path[i + 1]]['weight'] for i in range(len(best_path) - 1))
            break
    return best_path if current == end else None, best_cost

def tabu_search_with_sampling(graph, start, end, max_iterations=100, tabu_size=10, sample_size=3):
    current = start
    best_path = [current]
    best_cost = float('inf')
    tabu_list = []
    for _ in range(max_iterations):
        neighbors = list(graph.neighbors(current))
        neighbors = [n for n in neighbors if n not in tabu_list]
        if not neighbors:
            break
        sampled_neighbors = random.sample(neighbors, min(sample_size, len(neighbors)))
        next_node = min(sampled_neighbors, key=lambda n: graph[current][n]['weight'])
        best_path.append(next_node)
        tabu_list.append(current)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)
        current = next_node
        if current == end:
            best_cost = sum(graph[best_path[i]][best_path[i + 1]]['weight'] for i in range(len(best_path) - 1))
            break
    return best_path if current == end else None, best_cost


def main():
    # load csv 
    graph = load_data("connection_graph.csv") 
    # declare start and end 
    start, end = "FAT", "Berenta"
    # run the algorithm from the start
    # start timer 
    djkstrasTime = stopWatch("djkstra")
    djkstrasTime.start()
    dist, prev = dijkstra(graph, start)
    path = get_path(prev, start, end)
    djkstrasTime.stop()
    print(f"Route: {path}, Time: {dist.get(end, '∞')} sec")
    
    # Run A* algorithm
    aStarTimer = stopWatch("A star")
    aStarTimer.start()
    a_star_path = a_star(graph, start, end)
    aStarTimer.stop()
    if a_star_path:
        print(f"A* Route: {a_star_path}")
    else:
        print("No route found using A*.")
    
    # Visualize A* path
    def show_a_star_path(graph, path):
        if not path:
            print("No path to visualize for A*.")
            return
        subgraph = graph.subgraph(path)
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(subgraph, seed=random.randint(0, 100))
        nx.draw(subgraph, pos, with_labels=True, node_color="lightgreen", edge_color="blue", node_size=700, font_size=10)
        nx.draw_networkx_edges(subgraph, pos, edgelist=list(zip(path, path[1:])), edge_color="blue", width=2.5)
        nx.draw_networkx_nodes(subgraph, pos, nodelist=path, node_color="orange", node_size=700)
        plt.title("A* Path Visualization")
        plt.show()

    show_a_star_path(graph, a_star_path)
    # End the timer
    
    # Run Tabu Search algorithm
    # (it doesn't work at all for some odd reason)
    # tabu search assisted heavily by AI ~60%
    tabuTimer = stopWatch("Tabu Search")
    tabuTimer.start()
    tabu_path, tabu_cost = tabu_search(graph, start, end)
    tabuTimer.stop()
    if tabu_path:
        print(f"Tabu Search Route: {tabu_path}, Cost: {tabu_cost}")
    else:
        print("No route found using Tabu Search.")

    # Visualize Tabu Search path
    def show_tabu_path(graph, path):
        if not path:
            print("No path to visualize for Tabu Search.")
            return
        subgraph = graph.subgraph(path)
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(subgraph, seed=random.randint(0, 100))
        nx.draw(subgraph, pos, with_labels=True, node_color="lightcoral", edge_color="purple", node_size=700, font_size=10)
        nx.draw_networkx_edges(subgraph, pos, edgelist=list(zip(path, path[1:])), edge_color="purple", width=2.5)
        nx.draw_networkx_nodes(subgraph, pos, nodelist=path, node_color="pink", node_size=700)
        plt.title("Tabu Search Path Visualization")
        plt.show()

    show_tabu_path(graph, tabu_path)
    # Calculate elapsed time
    # djkstras
    # 10.013 seconds
    # 9.59 seconds
    # 9.58
    # 9.70
    #9.72 seconds average execution time after 4 runs

    show_graph(graph, path)

# so it can't be run as a module 
if __name__ == "__main__":
    main()
