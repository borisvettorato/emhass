#!/usr/bin/env python
"""Unit tests for emhass.persistence's keep_previous backup mechanism."""

import asyncio
import logging
import pathlib
import tempfile
import unittest

from emhass.persistence import load_json_blob, load_pickle_blob, save_json_blob, save_pickle_blob


class TestPersistenceKeepPrevious(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.emhass_conf = {"data_path": pathlib.Path(self._tmpdir.name)}
        self.logger = logging.getLogger("test_persistence")

    def tearDown(self):
        self._tmpdir.cleanup()

    async def test_save_json_blob_keep_previous_backs_up_old_content(self):
        """keep_previous=True must leave a "<name>.previous.json" sibling
        holding the OLD content after a second save overwrites the file -
        the mechanism refit_rc_model/refit_arx_model/
        refit_hybrid_heatpump_model rely on so a bad-but-still-gate-clearing
        refit isn't unrecoverable."""
        await save_json_blob(self.emhass_conf, "thing.json", {"version": 1}, self.logger, keep_previous=True)
        await save_json_blob(self.emhass_conf, "thing.json", {"version": 2}, self.logger, keep_previous=True)

        current = await load_json_blob(self.emhass_conf, "thing.json", self.logger)
        previous = await load_json_blob(self.emhass_conf, "thing.previous.json", self.logger)
        self.assertEqual(current, {"version": 2})
        self.assertEqual(previous, {"version": 1})

    async def test_save_json_blob_keep_previous_no_backup_on_first_save(self):
        """No ".previous.json" must appear when there was nothing to back
        up yet (a room/house's very first deploy)."""
        await save_json_blob(self.emhass_conf, "thing.json", {"version": 1}, self.logger, keep_previous=True)

        previous_path = pathlib.Path(self.emhass_conf["data_path"]) / "thing.previous.json"
        self.assertFalse(previous_path.exists())

    async def test_save_json_blob_default_does_not_keep_previous(self):
        """Without keep_previous (the default), a second save must NOT
        leave a backup - today's exact existing behavior, unchanged for
        every non-deploy-time caller."""
        await save_json_blob(self.emhass_conf, "thing.json", {"version": 1}, self.logger)
        await save_json_blob(self.emhass_conf, "thing.json", {"version": 2}, self.logger)

        previous_path = pathlib.Path(self.emhass_conf["data_path"]) / "thing.previous.json"
        self.assertFalse(previous_path.exists())

    async def test_save_json_blob_concurrent_writes_to_same_file_all_succeed(self):
        """Regression guard: two overlapping save_json_blob calls to the
        SAME destination (e.g. thermal_comfort.html's schedule-save firing
        on every mouse-move during a drag, faster than one round-trip)
        must NOT race on a shared temp filename - each call gets its own
        unique tmp file (see _unique_tmp_path), so whichever finishes
        first renaming its own tmp file away never leaves the other
        looking for a tmp file that's already gone. Before this fix, one
        of these would intermittently fail with
        FileNotFoundError/[Errno 2] on os.replace."""
        results = await asyncio.gather(
            *[
                save_json_blob(self.emhass_conf, "concurrent.json", {"writer": i}, self.logger)
                for i in range(20)
            ]
        )
        self.assertTrue(all(results), f"some concurrent saves failed: {results}")
        # The file must be in a valid, fully-written state from exactly one
        # of the writers - never truncated/corrupted by two writers'
        # temp files colliding mid-write.
        final = await load_json_blob(self.emhass_conf, "concurrent.json", self.logger)
        self.assertIn(final, [{"writer": i} for i in range(20)])

    async def test_save_pickle_blob_keep_previous_backs_up_old_content(self):
        """Same keep_previous behavior for save_pickle_blob (used for the
        ARX-model and hybrid-heatpump .pkl model files)."""
        await save_pickle_blob(self.emhass_conf, "thing.pkl", {"version": 1}, self.logger, keep_previous=True)
        await save_pickle_blob(self.emhass_conf, "thing.pkl", {"version": 2}, self.logger, keep_previous=True)

        current = await load_pickle_blob(self.emhass_conf, "thing.pkl", self.logger)
        previous = await load_pickle_blob(self.emhass_conf, "thing.previous.pkl", self.logger)
        self.assertEqual(current, {"version": 2})
        self.assertEqual(previous, {"version": 1})


if __name__ == "__main__":
    unittest.main()
