"""RSI-PDE Bench splits: sizes, opaque ids, no overlap with any other case, hidden data kept evaluator-only."""

import collections
import unittest

from pde_rsi import expanded as X
from pde_rsi import inventory as inv
from pde_rsi import pdebench as pb
from pde_rsi import pdebench_ext as px
from pde_rsi.bench import splits as S
from pde_rsi.bench.spec import PER_TEMPLATE, TRACKS
from pde_rsi.storage import PROJECT_ROOT


class BenchSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = {(track, split): S.tasks(track, split) for track in TRACKS for split in ("train", "val", "test")}

    def test_every_template_appears_per_template_times(self):
        for (track, split), items in self.tasks.items():
            template = (lambda t: t.spec["problem"]) if track != "inventory" else (lambda t: t.spec["family_key"])
            counts = collections.Counter(template(t) for t in items)
            self.assertEqual(set(counts), set(S.templates(track)), (track, split))
            self.assertEqual(set(counts.values()), {PER_TEMPLATE[split]}, (track, split))

    def test_ids_reveal_no_template(self):
        for (track, split), items in self.tasks.items():
            for task in items:  # track, split and a hash only
                self.assertRegex(task.task_id, rf"^bench-{track}-{split}-[0-9a-f]{{10}}$")
        cases = pb.make_cases("bench-val")
        self.assertEqual([c.case_id for c in cases], [f"case-bv{i:02d}" for i in range(20)])
        self.assertEqual({c.truth.split for c in cases}, {"development"})

    def test_no_case_repeats_another(self):
        keys = [X._key(t.spec) for items in self.tasks.values() for t in items]
        self.assertEqual(len(set(keys)), len(keys))
        others = {X._key(c.truth.spec) for c in X.tex_cases()+X.ext_cases()+X.inventory_cases()}
        others |= {X._key(c.truth.spec) for split in pb.SPLITS+pb.SEALED_SPLITS[1:] for c in pb.make_cases(split)}
        others |= {X._key(c.truth.spec) for split in px.SPLITS for c in px.make_cases(split)}
        others |= {X._key(inv.make_spec(f, s)) for f in inv.FAMILIES for s in inv.SPLITS+inv.SEALED_SPLITS[1:]}
        self.assertFalse(set(keys) & others)

    def test_hidden_test_is_evaluator_only(self):
        private = PROJECT_ROOT/"evaluator_private"
        for track in ("ext", "inventory"):
            module = px if track == "ext" else inv
            for task in self.tasks[(track, "test")]:
                self.assertTrue(module.reference_path(task).is_relative_to(private))
            for task in self.tasks[(track, "train")]:
                self.assertFalse(module.reference_path(task).is_relative_to(private))
        hidden = {t.spec.get("dataset_instance") for t in self.tasks[("inventory", "test")]} - {None}
        public = {t.spec.get("dataset_instance") for s in ("train", "val") for t in self.tasks[("inventory", s)]}
        self.assertTrue(hidden and not hidden & public and min(hidden) >= 200)
        self.assertIn(S.secret(), S.version("test"))
        self.assertNotIn(S.secret(), S.version("train")+S.version("val"))


if __name__ == "__main__":
    unittest.main()
