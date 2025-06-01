import networkx as nx

class SwellVisualizer:
    def create_propagation_network(tracker: SwellTracker):
        G = nx.DiGraph()
        
        # Add buoy nodes
        for bid, buoy in tracker.buoys.items():
            G.add_node(bid, 
                pos=(buoy.lon, buoy.lat),
                type='buoy')
        
        # Add propagation edges
        for (source, target), paths in tracker.propagation_graph.items():
            avg_delay = np.mean([p[2] for p in paths])
            G.add_edge(source, target, 
                      delay=avg_delay,
                      paths=paths)
        
        return G

    def plot_network(G, current_time=None):
        """Plot propagation network with active swells"""
        # Custom Folium map visualization
        return map