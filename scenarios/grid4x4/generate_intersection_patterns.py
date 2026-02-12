import random
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import itertools


def parse_network(net_file: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Parse .net.xml file to extract all possible routes through the intersection.
    
    Returns:
        Dictionary mapping incoming edges to list of (outgoing_edge, direction) tuples
    """
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    # Find all connections (excluding internal connections)
    connections = defaultdict(list)
    
    for conn in root.findall('.//connection'):
        from_edge = conn.get('from')
        to_edge = conn.get('to')
        direction = conn.get('dir', 'unknown')  # r=right, s=straight, l=left
        
        # Skip internal connections (those with ':' in the name)
        if from_edge and to_edge and ':' not in from_edge and ':' not in to_edge:
            connections[from_edge].append((to_edge, direction))
    
    return connections


def group_routes_by_direction(connections: Dict[str, List[Tuple[str, str]]]) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Group routes by incoming direction and assign turning probabilities.
    
    Returns:
        Dictionary with direction labels as keys, and lists of (from_edge, to_edge, turn_type) tuples
    """
    # Get all unique incoming edges (these represent different approach directions)
    incoming_edges = list(connections.keys())
    
    routes_by_approach = {}
    
    for idx, from_edge in enumerate(incoming_edges):
        direction_label = f"DIR_{chr(65+idx)}"  # DIR_A, DIR_B, DIR_C, DIR_D
        routes_by_approach[direction_label] = [
            (from_edge, to_edge, turn_type) 
            for to_edge, turn_type in connections[from_edge]
        ]
    
    return routes_by_approach


def get_turn_probability(turn_type: str) -> float:
    """Assign probability based on turn type."""
    turn_probs = {
        's': 0.60,  # Straight - most common
        'r': 0.25,  # Right turn
        'l': 0.15,  # Left turn
        'unknown': 0.33,  # Equal if unknown
    }
    return turn_probs.get(turn_type, 0.33)


def select_most_significant_patterns(num_directions: int = 4, max_patterns: int = 15) -> List[Tuple]:
    """
    Select the most significant traffic patterns based on combinations of intensity multipliers.
    
    Args:
        num_directions: Number of approach directions
        max_patterns: Maximum number of patterns to generate
        
    Returns:
        List of tuples containing intensity multipliers for each direction and a name
    """
    # Define intensity levels
    intensities = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    patterns = []
    
    # 1. Balanced scenario (all equal)
    balanced = tuple([1.0] * num_directions)
    patterns.append((*balanced, "balanced"))
    
    # 2. Single direction dominant scenarios
    for dir_idx in range(num_directions):
        multipliers = [0.5] * num_directions
        multipliers[dir_idx] = 2.5  # One direction heavy
        patterns.append((*multipliers, f"dir_{chr(65+dir_idx)}_heavy"))
    
    # 3. Two directions dominant (if 4+ directions)
    if num_directions >= 4:
        # Opposite directions (0-2, 1-3 for 4-way)
        for i in range(min(2, num_directions // 2)):
            multipliers = [0.5] * num_directions
            opposite_idx = (i + num_directions // 2) % num_directions
            multipliers[i] = 2.0
            multipliers[opposite_idx] = 2.0
            patterns.append((*multipliers, f"dirs_{chr(65+i)}{chr(65+opposite_idx)}_heavy"))
    
    # 4. Adjacent directions dominant
    if num_directions >= 3:
        for i in range(num_directions):
            next_idx = (i + 1) % num_directions
            multipliers = [0.5] * num_directions
            multipliers[i] = 1.8
            multipliers[next_idx] = 1.8
            patterns.append((*multipliers, f"dirs_{chr(65+i)}{chr(65+next_idx)}_heavy"))
    
    # 5. Extreme imbalance scenarios
    for dir_idx in range(min(2, num_directions)):  # Just first 2 to avoid too many
        multipliers = [0.3] * num_directions
        multipliers[dir_idx] = 3.0
        patterns.append((*multipliers, f"dir_{chr(65+dir_idx)}_extreme"))
    
    # 6. Gradual variations
    if num_directions == 4:
        patterns.append((2.0, 1.5, 0.8, 0.5, "gradient_descending"))
        patterns.append((0.5, 0.8, 1.5, 2.0, "gradient_ascending"))
    
    # Limit to max_patterns, prioritizing balanced and single-direction scenarios
    if len(patterns) > max_patterns:
        # Keep first few patterns (balanced + single direction) and sample from the rest
        patterns = patterns[:num_directions + 1] + patterns[num_directions + 1:max_patterns]
    
    return patterns[:max_patterns]


def generate_scenarios(net_file: str,
                      output_file: str,
                      segment_length: int = 3600,
                      base_veh_per_hour: int = 1200,
                      seed: int = 42,
                      max_patterns: int = 15):
    """
    Generate route file with multiple traffic patterns in sequence.
    
    Args:
        net_file: Path to .net.xml file
        output_file: Name of output .rou.xml file
        segment_length: Duration of each pattern segment in seconds
        base_veh_per_hour: Base vehicle generation rate
        seed: Random seed
        max_patterns: Maximum number of patterns to generate
    """
    random.seed(seed)
    
    # Parse network to get all possible routes
    print(f"\nAnalyzing network: {net_file}")
    connections = parse_network(net_file)
    
    if not connections:
        print("ERROR: No valid connections found in network file!")
        return 0
    
    routes_by_approach = group_routes_by_direction(connections)
    
    # Print network analysis
    print(f"\nNetwork Analysis:")
    print(f"  Total approach directions: {len(routes_by_approach)}")
    for direction, routes in routes_by_approach.items():
        print(f"  {direction}: {len(routes)} possible routes")
        for from_edge, to_edge, turn_type in routes:
            print(f"    {from_edge} -> {to_edge} ({turn_type})")
    
    # Create route definitions for XML
    route_definitions = []
    route_index = 0
    for direction, routes in routes_by_approach.items():
        for from_edge, to_edge, turn_type in routes:
            turn_prob = get_turn_probability(turn_type)
            route_definitions.append({
                'id': route_index,
                'edges': f"{from_edge} {to_edge}",
                'from_edge': from_edge,
                'to_edge': to_edge,
                'direction': direction,
                'turn_type': turn_type,
                'base_prob': turn_prob,
            })
            route_index += 1
    
    if not route_definitions:
        print("ERROR: No routes to generate!")
        return 0
    
    # Select most significant patterns
    num_directions = len(routes_by_approach)
    test_cases = select_most_significant_patterns(num_directions, max_patterns)
    
    total_duration = len(test_cases) * segment_length
    
    print(f"\nGenerating Combined Evaluation File...")
    print(f"  Patterns: {len(test_cases)}")
    print(f"  Segment length: {segment_length}s ({segment_length/60:.1f} min)")
    print(f"  Total duration: {total_duration}s ({total_duration/60:.1f} min = {total_duration/3600:.1f} hours)")
    
    # Generate XML file
    with open(output_file, "w") as routes:
        routes.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        routes.write('<!-- Auto-generated Traffic Pattern Library -->\n')
        routes.write(f'<!-- Total Duration: {total_duration}s ({total_duration/3600:.1f} hours) -->\n')
        routes.write(f'<!-- Network: {Path(net_file).name} -->\n')
        routes.write('<routes>\n')
        routes.write('    <vType id="passenger" accel="0.8" decel="4.5" length="5.0" '
                    'minGap="2.5" maxSpeed="13.89" vClass="passenger" '
                    'speedFactor="1.0" speedDev="0.1" '
                    'jmIgnoreJunctionFoeProb="1.0" jmIgnoreKeepClearTime="-1" />\n\n')
        
        # Write route definitions
        for route_def in route_definitions:
            routes.write(f'    <route id="route_{route_def["id"]}" '
                        f'edges="{route_def["edges"]}" />\n')
        routes.write('\n')
        
        # Map direction labels to indices
        direction_list = list(routes_by_approach.keys())
        
        # Generate vehicles for each pattern segment
        total_vehicles = 0
        
        for segment_idx, pattern in enumerate(test_cases):
            # Extract multipliers and name
            *multipliers, name = pattern
            multipliers_dict = {direction_list[i]: mult for i, mult in enumerate(multipliers)}
            
            segment_start = segment_idx * segment_length
            segment_end = segment_start + segment_length
            
            mult_str = " ".join([f"{dir}={mult:.1f}x" for dir, mult in multipliers_dict.items()])
            routes.write(f'\n    <!-- Segment {segment_idx + 1}: {name.upper()} '
                        f'({mult_str}) | {segment_start}-{segment_end}s -->\n')
            
            t = segment_start
            veh_id = total_vehicles
            segment_veh_count = 0
            
            avg_arrival_rate = base_veh_per_hour / 3600.0
            
            while t < segment_end:
                inter_arrival_time = random.expovariate(avg_arrival_rate)
                t += inter_arrival_time
                
                if t >= segment_end:
                    break
                
                # Build weighted route selection based on direction multipliers
                weights = []
                for route_def in route_definitions:
                    direction = route_def['direction']
                    direction_weight = multipliers_dict[direction]
                    turning_prob = route_def['base_prob']
                    weights.append(direction_weight * turning_prob)
                
                # Select route
                total_weight = sum(weights)
                if total_weight == 0:
                    continue
                    
                probabilities = [w / total_weight for w in weights]
                selected_route_idx = random.choices(range(len(route_definitions)), 
                                                   weights=probabilities)[0]
                
                selected_route = route_definitions[selected_route_idx]
                routes.write(f'    <vehicle id="veh_{veh_id}" type="passenger" '
                            f'route="route_{selected_route["id"]}" depart="{t:.2f}" />\n')
                veh_id += 1
                segment_veh_count += 1
            
            total_vehicles = veh_id
            print(f"  ✓ Segment {segment_idx + 1}: {name:25s} | {segment_veh_count:4d} vehicles | "
                  f"{segment_start:5d}-{segment_end:5d}s")
        
        routes.write('</routes>\n')
    
    print(f"\n✓ File generated: {output_file}")
    print(f"✓ Total vehicles: {total_vehicles}")
    if total_duration > 0:
        print(f"✓ Average: {total_vehicles * 3600 / total_duration:.0f} veh/h")
    
    return total_vehicles


def create_sumo_config(net_file: str, route_file: str, output_file: str, end_time: int = 10000):
    """
    Create a SUMO configuration file.
    
    Args:
        net_file: Path to network file
        route_file: Path to route file
        output_file: Path to output .sumocfg file
        end_time: Simulation end time in seconds
    """
    config_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="{Path(net_file).name}"/>
        <route-files value="{Path(route_file).name}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{end_time}"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
        <time-to-teleport.highways value="-1"/>
        <time-to-teleport.disconnected value="-1"/>
        <time-to-teleport.remove value="false"/>
        <max-depart-delay value="-1"/>
        <ignore-junction-blocker value="-1"/>
        <ignore-route-errors value="true"/>
        <collision.action value="none"/>
        <collision.check-junctions value="false"/>
        <collision.mingap-factor value="0"/>
        <lanechange.duration value="10"/>
        <eager-insert value="true"/>
        <emergencydecel.warning-threshold value="0"/>
    </processing>
    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
        <no-warnings value="true"/>
    </report>
</configuration>
"""
    
    with open(output_file, 'w') as f:
        f.write(config_content)
    
    print(f"  ✓ Created config: {Path(output_file).name}")


def generate_for_intersection(intersection_dir: str, 
                              segment_length_train: int = 1800,
                              segment_length_eval: int = 600,
                              base_veh_per_hour: int = 1200):
    """
    Generate both training and evaluation route files for a specific intersection.
    
    Args:
        intersection_dir: Path to intersection directory (e.g., 'scenarios/berlin-small/A')
        segment_length_train: Segment duration for training (seconds)
        segment_length_eval: Segment duration for evaluation (seconds)
        base_veh_per_hour: Base vehicle generation rate
    """
    intersection_path = Path(intersection_dir)
    intersection_name = intersection_path.name.lower()
    
    # Find the .net.xml file - try pattern match first, then look for any .net.xml
    net_file = intersection_path / f"{intersection_name}.net.xml"
    if not net_file.exists():
        # Try to find any .net.xml file in the directory
        net_files = list(intersection_path.glob("*.net.xml"))
        if net_files:
            net_file = net_files[0]
    
    if not net_file.exists():
        print(f"ERROR: Network file not found: {net_file}")
        return
    
    print(f"\n{'='*70}")
    print(f"Generating patterns for Intersection {intersection_name.upper()}")
    print(f"{'='*70}")
    
    # Generate training file (using safe filename to avoid overwriting)
    train_file = intersection_path / "train_generated.rou.xml"
    generate_scenarios(
        net_file=str(net_file),
        output_file=str(train_file),
        segment_length=segment_length_train,
        base_veh_per_hour=base_veh_per_hour,
        seed=42,
        max_patterns=15
    )
    
    # Generate evaluation file (using safe filename to avoid overwriting)
    eval_file = intersection_path / "eval_generated.rou.xml"
    generate_scenarios(
        net_file=str(net_file),
        output_file=str(eval_file),
        segment_length=segment_length_eval,
        base_veh_per_hour=base_veh_per_hour,
        seed=999,
        max_patterns=12  # Fewer patterns for quicker eval
    )
    
    # Generate SUMO config files
    print(f"\nGenerating SUMO configuration files...")
    train_config = intersection_path / "train_generated.sumocfg"
    create_sumo_config(
        net_file=str(net_file),
        route_file=str(train_file),
        output_file=str(train_config),
        end_time=len(select_most_significant_patterns(
            parse_network(str(net_file)).__len__(), 15
        )) * segment_length_train
    )
    
    eval_config = intersection_path / "eval_generated.sumocfg"
    create_sumo_config(
        net_file=str(net_file),
        route_file=str(eval_file),
        output_file=str(eval_config),
        end_time=len(select_most_significant_patterns(
            parse_network(str(net_file)).__len__(), 12
        )) * segment_length_eval
    )


if __name__ == "__main__":
    # Configuration for grid4x4 scenario
    base_path = Path(__file__).parent
    intersection_dir = base_path  # Use current directory (scenarios/grid4x4)

    generate_for_intersection(
        intersection_dir=str(intersection_dir),
        segment_length_train=1800,  # 30 minutes per pattern
        segment_length_eval=600,     # 10 minutes per pattern
        base_veh_per_hour=2500      # Adjust to 1200 for grid network
    )

    print(f"\n{'='*70}")
    print("Grid4x4 patterns generated!")
    print(f"{'='*70}")
