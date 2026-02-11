"""Reward function implementations"""


def reward_minimize_queue(ts):
    """
    Reward based on queue length.
    Directly penalizes the number of stopped cars.
    
    Args:
        ts: TrafficSignal instance
        
    Returns:
        float: Negative queue length
    """
    return -float(ts.get_total_queued())


def reward_minimize_max_queue(ts):
    """
    Reward based on maximum queue length across all lanes.
    Penalizes the longest queue (worst-case scenario).
    This encourages balancing traffic across lanes.
    
    Args:
        ts: TrafficSignal instance
        
    Returns:
        float: Negative maximum queue length
    """
    # Get queue length for each lane
    lanes = ts.lanes
    max_queue = 0
    
    for lane in lanes:
        queue_length = ts.sumo.lane.getLastStepHaltingNumber(lane)
        max_queue = max(max_queue, queue_length)
    
    return -float(max_queue)


def reward_vidali_waiting_time(ts):
    """
    Reward based on accumulated waiting time.
    Matches the 'Deep-QLearning-Agent' repository logic.
    
    Args:
        ts: TrafficSignal instance
        
    Returns:
        float: Negative sum of accumulated waiting time
    """
    # Get dictionary of wait times {lane_id: wait_time}
    wait_times = ts.get_accumulated_waiting_time_per_lane()
    
    # Sum them all up and negate
    total_wait = sum(wait_times)
    
    return -float(total_wait)
