"""SUMO XML parsing helpers shared by training/evaluation code."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def detect_route_duration_seconds(route_file_path: str):
    """
    Detect route duration in seconds from a SUMO route file.
    Priority:
    1) "Total Duration: <N>s" comment anywhere in the file
    2) max(<vehicle depart>) + 1 second fallback
    """
    xml_text = Path(route_file_path).read_text(encoding="utf-8")

    match = re.search(r"Total Duration:\s*(\d+)\s*s", xml_text)
    if match:
        return int(match.group(1)), "comment"

    root = ET.fromstring(xml_text)
    max_depart = 0.0
    for vehicle in root.findall("vehicle"):
        depart = vehicle.get("depart")
        if depart is None:
            continue
        try:
            max_depart = max(max_depart, float(depart))
        except ValueError:
            continue

    if max_depart > 0:
        return int(np.ceil(max_depart)) + 1, "max_depart"

    return None, None


def read_summary_arrived(summary_path):
    if not summary_path:
        return None
    try:
        root = ET.parse(summary_path).getroot()
        steps = root.findall("step")
        if not steps:
            return None
        return int(float(steps[-1].attrib.get("arrived", 0)))
    except Exception:
        return None


def read_summary_metrics(summary_path):
    """Read robust system metrics from SUMO summary output."""
    if not summary_path:
        return {}
    try:
        root = ET.parse(summary_path).getroot()
        steps = root.findall("step")
        if not steps:
            return {}
        last = steps[-1].attrib
        mean_wait_series = [float(s.attrib.get("meanWaitingTime", 0.0)) for s in steps]
        mean_speed_series = [float(s.attrib.get("meanSpeed", 0.0)) for s in steps]
        halting_series = [float(s.attrib.get("halting", 0.0)) for s in steps]
        return {
            "summary_total_arrived": int(float(last.get("arrived", 0))),
            "summary_total_teleports": int(float(last.get("teleports", 0))),
            "summary_total_collisions": int(float(last.get("collisions", 0))),
            "summary_total_inserted": int(float(last.get("inserted", 0))),
            "summary_total_running_end": int(float(last.get("running", 0))),
            "summary_total_waiting_end": int(float(last.get("waiting", 0))),
            "summary_mean_waiting_time_end": float(last.get("meanWaitingTime", 0.0)),
            "summary_mean_speed_end": float(last.get("meanSpeed", 0.0)),
            "summary_halting_end": float(last.get("halting", 0.0)),
            "summary_mean_waiting_time_avg": float(np.mean(mean_wait_series)),
            "summary_mean_speed_avg": float(np.mean(mean_speed_series)),
            "summary_halting_avg": float(np.mean(halting_series)),
        }
    except Exception:
        return {}


def read_tripinfo_metrics(tripinfo_path):
    """Read per-trip metrics from SUMO tripinfo output."""
    if not tripinfo_path:
        return {}
    try:
        root = ET.parse(tripinfo_path).getroot()
        trips = root.findall("tripinfo")
        if not trips:
            return {
                "tripinfo_completed_trips": 0,
                "tripinfo_mean_duration": 0.0,
                "tripinfo_p90_duration": 0.0,
                "tripinfo_mean_waiting_time": 0.0,
                "tripinfo_p90_waiting_time": 0.0,
            }
        durations = np.array([float(t.attrib.get("duration", 0.0)) for t in trips], dtype=float)
        waits = np.array([float(t.attrib.get("waitingTime", 0.0)) for t in trips], dtype=float)
        return {
            "tripinfo_completed_trips": int(len(trips)),
            "tripinfo_mean_duration": float(np.mean(durations)),
            "tripinfo_p90_duration": float(np.percentile(durations, 90)),
            "tripinfo_mean_waiting_time": float(np.mean(waits)),
            "tripinfo_p90_waiting_time": float(np.percentile(waits, 90)),
        }
    except Exception:
        return {}
