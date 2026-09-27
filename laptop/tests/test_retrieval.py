import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lanetalk import retrieval
from lanetalk.events import Event


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = mock.patch.dict(os.environ, {"LANETALK_TIPS_DB": str(Path(self.tmp.name) / "tips.db")})
        env.start()
        self.addCleanup(env.stop)

    def test_direction_specific_tip_ranks_first_and_other_direction_excluded(self):
        tips = retrieval.retrieve(Event("lane_drift", 0.9, details={"direction": "left"}))
        self.assertEqual(tips[0], "Drifting left. Steer gently back to center.")
        self.assertFalse(any("right" in tip for tip in tips))
        self.assertEqual(len(tips), 3)

    def test_vehicle_class_and_top_k(self):
        tips = retrieval.retrieve(Event("vehicle_detected", 0.8, details={"class": "truck"}), top_k=2)
        self.assertEqual(tips, ["Truck ahead. Leave extra room.", "Vehicle ahead. Keep a safe following distance."])

    def test_accepts_kind_string_and_unknown_kind_is_empty(self):
        self.assertEqual(len(retrieval.retrieve("pedestrian_detected")), 3)
        self.assertEqual(retrieval.retrieve("unknown_event"), [])
        self.assertEqual(retrieval.retrieve("lane_drift", top_k=0), [])

    def test_seed_edits_are_picked_up(self):
        seed = Path(self.tmp.name) / "seed.json"
        seed.write_text(json.dumps([{"kind": "lane_drift", "text": "old"}]))
        with mock.patch.object(retrieval, "SEED_PATH", seed):
            self.assertEqual(retrieval.retrieve("lane_drift"), ["old"])
            seed.write_text(json.dumps([{"kind": "lane_drift", "text": "new"}]))
            self.assertEqual(retrieval.retrieve("lane_drift"), ["new"])


if __name__ == "__main__":
    unittest.main()
