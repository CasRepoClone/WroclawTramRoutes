#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

start_time = time.time() # timer for performance testing

# roughly 30% assisted by AI
def time_to_seconds(time_str):
    # didnt use date time as it was being broken 
    h, m, s = map(int, time_str.split(':'))
    # split the time into hours, minutes and second then return in seconds 
    return h * 3600 + m * 60 + s

def heuristic(a, b):
    # A* algorithm
    return g(a) + h(b)

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

def main():
    # load csv 
    graph = load_data("Project A/connection_graph.csv") 
    # declare start and end 
    start, end = "FAT", "Berenta"
    # run the algorithm from the start
    dist, prev = dijkstra(graph, start)
    path = get_path(prev, start, end)
    print(f"Route: {path}, Time: {dist.get(end, '∞')} sec")


    # End the timer
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time} seconds")
    # 10.013 seconds
    # 9.59 seconds
    # 9.58
    # 9.70
    #9.72 seconds average execution time after 4 runs

    show_graph(graph, path)

# so it can't be run as a module 
if __name__ == "__main__":
    main()
