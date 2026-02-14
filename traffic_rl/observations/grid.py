"""Grid-based observation function"""

import numpy as np
from sumo_rl.environment.observations import ObservationFunction
from gymnasium import spaces


class GridObservationFunction(ObservationFunction):
    """
    Grid observation function for traffic lights.
    
    Replicates the 'Deep-QLearning-Agent' grid observation approach.
    Divides incoming lanes into binary cells (0 or 1) indicating vehicle presence,
    and appends one-hot encoding of current traffic signal phase.
    
    Args:
        ts: TrafficSignal instance
        num_cells: Number of cells to divide each lane into (default: 10)
    """
    
    def __init__(self, ts, num_cells=10):
        super().__init__(ts)
        self.num_cells = num_cells

    def observation_space(self):
        """
        Compute the observation space.
        
        Called by TrafficSignal after it has been fully initialized.
        
        Returns:
            spaces.Box: Observation space with size (num_lanes * num_cells + num_phases)
        """
        self.lane_ids = self.ts.lanes
        
        # Calculate observation space size:
        # (Number of Lanes * Num Cells) + (Number of Green Phases)
        obs_size = (len(self.lane_ids) * self.num_cells) + self.ts.num_green_phases
        return spaces.Box(low=0, high=1, shape=(obs_size,), dtype=np.float32)

    def __call__(self):
        """
        Generate the observation vector.
        
        Returns:
            np.ndarray: Observation vector with vehicle positions and phase encoding
        """
        # 1. Get Vehicle Positions in Grid
        grid = []
        
        # Ensure lane_ids are available
        if not hasattr(self, 'lane_ids'):
            self.lane_ids = self.ts.lanes

        for lane_id in self.lane_ids:
            # Get lane length
            length = self.ts.sumo.lane.getLength(lane_id)
            cell_length = length / self.num_cells
            
            # Create empty cells for this lane
            lane_cells = [0.0] * self.num_cells
            
            # Find where cars are
            vehicles = self.ts.sumo.lane.getLastStepVehicleIDs(lane_id)
            
            for veh in vehicles:
                # Get distance from start of lane
                pos = self.ts.sumo.vehicle.getLanePosition(veh)
                
                # Map position to cell index (0 to num_cells-1)
                # In SUMO, 0 is start of lane, length is the intersection
                cell_idx = int(pos / cell_length)
                
                cell_idx = min(cell_idx, self.num_cells - 1)
                lane_cells[cell_idx] = 1.0  # Mark occupied
            
            grid.extend(lane_cells)

        # 2. Add Phase Info (One-Hot Encoding)
        # ts.green_phase is the index of the current green phase
        phase_id = [1 if self.ts.green_phase == i else 0 
                   for i in range(self.ts.num_green_phases)]
        
        # Combine grid and phase
        observation = np.array(grid + phase_id, dtype=np.float32)
        return observation
