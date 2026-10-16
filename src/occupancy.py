"""Persistent seat registry and integration API for the webcam pipeline."""
from .geometry import assign_people
from .smoothing import SeatState


class OccupancyEngine:
    def __init__(self, seats, calibration=None, on_seconds=0.5,
                 off_seconds=1.0, max_gap_seconds=0.5, **matching_options):
        self.seats = list(seats)
        if len({s.seat_id for s in self.seats}) != len(self.seats):
            raise ValueError("Seat IDs must be unique")
        self.calibration = calibration
        self.matching_options = matching_options
        self.states = {s.seat_id: SeatState(on_seconds, off_seconds, max_gap_seconds)
                       for s in self.seats}

    def update(self, people, timestamp):
        """people=None: inference unavailable; []: inference ran, no people.

        Keep this instance across frames. Seats persist even when YOLO cannot
        see their boxes. Recreate/reconfigure explicitly when seats move.
        """
        matches = (assign_people(self.seats, people, self.calibration,
                                 **self.matching_options) if people is not None else None)
        rows = {}
        for seat in self.seats:
            index = matches[seat.seat_id] if matches is not None else None
            raw = index is not None if matches is not None else None
            stable = self.states[seat.seat_id].update(raw, timestamp)
            rows[seat.seat_id] = {"raw_occupied": raw, "occupied": stable,
                                  "person_index": index}
        return {"seats": rows, "total": len(rows),
                "available": sum(r["occupied"] is False for r in rows.values()),
                "occupied": sum(r["occupied"] is True for r in rows.values()),
                "unknown": sum(r["occupied"] is None for r in rows.values())}
