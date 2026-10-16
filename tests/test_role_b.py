import tempfile
import unittest
from pathlib import Path
from src.geometry import FloorCalibration, Person, Seat, assign_people
from src.occupancy import OccupancyEngine
from src.smoothing import SeatState


class RoleBTests(unittest.TestCase):
    def test_one_person_cannot_fill_two_seats(self):
        seats = [Seat("a", (0, 0, 10, 10)), Seat("b", (5, 0, 15, 10))]
        matches = assign_people(seats, [Person((0, 0, 15, 10))])
        self.assertEqual(sum(v is not None for v in matches.values()), 1)

    def test_floor_distance_rejects_overlap(self):
        h = FloorCalibration([[0, 0], [100, 0], [100, 100], [0, 100]], 2, 2)
        seats = [Seat("a", (0, 0, 100, 100), (50, 90))]
        people = [Person((0, 0, 100, 100), floor_anchor=(50, 10))]
        self.assertEqual(assign_people(seats, people)["a"], 0)
        self.assertIsNone(assign_people(seats, people, h)["a"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            h.save(path)
            self.assertEqual(FloorCalibration.load(path).to_floor((50, 50)), (1, 1))

    def test_bad_calibration(self):
        with self.assertRaises(ValueError):
            FloorCalibration([[0, 0], [1, 1], [2, 2], [3, 3]], 2, 2)

    def test_hysteresis_and_missing_evidence(self):
        state = SeatState(on_seconds=0.5, off_seconds=1, max_gap_seconds=0.6)
        self.assertIsNone(state.update(True, 0))
        self.assertTrue(state.update(True, 0.5))
        self.assertTrue(state.update(False, 0.6))
        self.assertTrue(state.update(None, 0.8))
        self.assertTrue(state.update(False, 1))
        self.assertTrue(state.update(False, 1.5))
        self.assertFalse(state.update(False, 2))
        with self.assertRaises(ValueError):
            state.update(True, 1)

    def test_long_gap_does_not_count_as_continuous_evidence(self):
        state = SeatState()
        state.update(True, 0)
        self.assertIsNone(state.update(True, 10))

    def test_engine_retains_seats_and_handles_detector_failure(self):
        engine = OccupancyEngine([Seat("a", (0, 0, 10, 10))])
        self.assertEqual(engine.update([], 0)["unknown"], 1)
        engine.update([], 0.5)
        self.assertEqual(engine.update([], 1)["available"], 1)
        result = engine.update(None, 1.1)
        self.assertEqual(result["total"], 1)
        self.assertIsNone(result["seats"]["a"]["raw_occupied"])
        self.assertEqual(result["available"], 1)


if __name__ == "__main__":
    unittest.main()
