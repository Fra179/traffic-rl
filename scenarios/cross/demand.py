import random
random.seed(42)

def generate_route_file(filename, total_time, veh_per_hour):
    # Your specific route definition list
    route_edges = [
        "t2c c2b", "t2c c2l", "t2c c2r",
        "b2c c2t", "b2c c2l", "b2c c2r",
        "l2c c2r", "l2c c2t", "l2c c2b",
        "r2c c2l", "r2c c2t", "r2c c2b"
    ]
    
    with open(filename, "w") as routes:
        routes.write('<routes>\n')
        
        # 1. Define the Vehicle Type
        routes.write('    <vType id="standard_car" accel="0.8" decel="4.5" length="5.0" />\n\n')
        
        # 2. Define the Routes in the XML
        # We give each route a unique ID (e.g., "route_0", "route_1")
        for i, edge_list in enumerate(route_edges):
            routes.write(f'    <route id="route_{i}" edges="{edge_list}" />\n')
            
        routes.write('\n')

        # Convert vehicles per hour to average arrival rate (lambda)
        avg_arrival_rate = veh_per_hour / 3600.0
        
        t = 0
        veh_id = 0
        
        # 3. Generate Vehicles
        while t < total_time:
            # Calculate next arrival time (Exponential Distribution)
            inter_arrival_time = random.expovariate(avg_arrival_rate)
            t += inter_arrival_time
            
            if t < total_time:
                # PICK A RANDOM ROUTE ID
                random_route_index = random.randint(0, len(route_edges) - 1)
                selected_route_id = f"route_{random_route_index}"
                
                # Write the vehicle entry
                routes.write(f'    <vehicle id="veh_{veh_id}" type="standard_car" '
                             f'route="{selected_route_id}" depart="{t:.2f}" />\n')
                veh_id += 1
                
        routes.write('</routes>')

# Usage: Generate traffic for 1 hour (3600s) with 1000 vehicles/hour
generate_route_file("cross.rou.xml", 3600, 2000) 

