"""Memory peaks must represent simultaneous whole-workflow samples."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from measure_workload_memory import summarize_samples


class MemorySummaryTests(unittest.TestCase):
    def test_simultaneous_peak_and_phase_separation(self):
        rows=[dict(seconds=t,phase=phase,device={'used':gpu},working_set_bytes=ram,
                   private_commit_bytes=commit,system_available_bytes=available)
              for t,phase,gpu,ram,commit,available in
              [(0,'idle',10,20,30,100),(.1,'image',90,60,50,80),(.35,'idol',70,80,90,60)]]
        result=summarize_samples(rows)
        self.assertEqual(result['overall']['process_tree_working_set_peak_bytes'],80)
        self.assertEqual(result['overall']['device_used_peak_bytes'],90)
        self.assertEqual(result['phases']['idle']['device_used_peak_bytes'],10)
        self.assertAlmostEqual(result['observed_sample_gap_seconds']['maximum'],.25)

    def test_absent_samples_refused(self):
        with self.assertRaises(ValueError):summarize_samples([])


if __name__=='__main__':unittest.main()
