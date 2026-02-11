import random


def generate_scenarios(output_file="routes.rou.xml", 
                                 segment_length=3600,
                                 base_veh_per_hour=1200,
                                 seed=42):
    """
    Generate ONE combined route file with all 9 evaluation patterns in sequence.
    
    This creates a single long episode where the agent experiences all 9 patterns:
    - 0-3600s: Balanced
    - 3600-7200s: NS slight
    - 7200-10800s: EW slight
    - ... etc for all 9 patterns
    
    Total duration: 9 * segment_length (default: 32400s = 9 hours)
    
    Args:
        output_file: Name of combined route file
        segment_length: Duration of each pattern segment in seconds
        base_veh_per_hour: Base vehicle rate
        seed: Random seed for reproducibility
        
    Returns:
        Total number of vehicles generated
    """
    random.seed(seed)
    
    # Key scenarios in order
    test_cases = [
        (1.0, 1.0, "balanced"),
        (1.5, 1.0, "ns_slight"),
        (1.0, 1.5, "ew_slight"),
        (2.0, 0.7, "ns_moderate"),
        (0.7, 2.0, "ew_moderate"),
        (2.5, 0.5, "ns_heavy"),
        (0.5, 2.5, "ew_heavy"),
        (3.0, 0.3, "ns_extreme"),
        (0.3, 3.0, "ew_extreme"),
    ]
    
    # Define routes
    route_definitions = [
        ("t2c c2b", "NS", 0.50),  ("t2c c2l", "NS", 0.25),  ("t2c c2r", "NS", 0.25),
        ("b2c c2t", "NS", 0.50),  ("b2c c2l", "NS", 0.25),  ("b2c c2r", "NS", 0.25),
        ("l2c c2r", "EW", 0.50),  ("l2c c2t", "EW", 0.25),  ("l2c c2b", "EW", 0.25),
        ("r2c c2l", "EW", 0.50),  ("r2c c2t", "EW", 0.25),  ("r2c c2b", "EW", 0.25),
    ]
    
    total_duration = len(test_cases) * segment_length
    
    print(f"\nGenerating Combined Evaluation File (All 9 Patterns Sequential)...")
    print(f"  Segment length: {segment_length}s ({segment_length/60:.1f} min)")
    print(f"  Total duration: {total_duration}s ({total_duration/60:.1f} min = {total_duration/3600:.1f} hours)")
    
    with open(output_file, "w") as routes:
        routes.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        routes.write('<!-- Combined Evaluation: All 9 Patterns Sequential -->\n')
        routes.write(f'<!-- Total Duration: {total_duration}s ({total_duration/3600:.1f} hours) -->\n')
        routes.write('<routes>\n')
        routes.write('    <vType id="passenger" accel="0.8" decel="4.5" length="5.0" '
                    'minGap="2.5" maxSpeed="13.89" vClass="passenger" />\n\n')
        
        # Write route definitions
        for i, (edges, direction, prob) in enumerate(route_definitions):
            routes.write(f'    <route id="route_{i}" edges="{edges}" />\n')
        routes.write('\n')
        
        # Generate vehicles for each pattern segment
        total_vehicles = 0
        
        for segment_idx, (ns_mult, ew_mult, name) in enumerate(test_cases):
            segment_start = segment_idx * segment_length
            segment_end = segment_start + segment_length
            
            routes.write(f'\n    <!-- Segment {segment_idx + 1}: {name.upper()} '
                        f'(NS={ns_mult}x, EW={ew_mult}x) | {segment_start}-{segment_end}s -->\n')
            
            direction_multipliers = {"NS": ns_mult, "EW": ew_mult}
            
            t = segment_start
            veh_id = total_vehicles
            segment_veh_count = 0
            
            avg_arrival_rate = base_veh_per_hour / 3600.0
            
            while t < segment_end:
                inter_arrival_time = random.expovariate(avg_arrival_rate)
                t += inter_arrival_time
                
                if t >= segment_end:
                    break
                
                # Build weighted route selection
                weights = []
                for i, (edges, direction, turning_prob) in enumerate(route_definitions):
                    direction_weight = direction_multipliers[direction]
                    weights.append(direction_weight * turning_prob)
                
                # Select route
                total_weight = sum(weights)
                probabilities = [w / total_weight for w in weights]
                selected_route_idx = random.choices(range(len(route_definitions)), 
                                                   weights=probabilities)[0]
                
                routes.write(f'    <vehicle id="veh_{veh_id}" type="passenger" '
                            f'route="route_{selected_route_idx}" depart="{t:.2f}" />\n')
                veh_id += 1
                segment_veh_count += 1
            
            total_vehicles = veh_id
            print(f"  ✓ Segment {segment_idx + 1}: {name:15s} | {segment_veh_count:4d} vehicles | "
                  f"{segment_start:5d}-{segment_end:5d}s")
        
        routes.write('</routes>\n')
    
    print(f"\n✓ Combined file generated: {output_file}")
    print(f"✓ Total vehicles: {total_vehicles}")
    print(f"✓ Average: {total_vehicles * 3600 / total_duration:.0f} veh/h")
    
    return total_vehicles


if __name__ == "__main__":
    # Generate combined evaluation file with all 9 patterns in sequence
    generate_scenarios(output_file="train.rou.xml", segment_length=1800, base_veh_per_hour=2000, seed=42)
    generate_scenarios(output_file="eval.rou.xml", segment_length=600, base_veh_per_hour=2000, seed=999)