"""
`webng setup` (weConvert.run) works from inside the sim folder; it must
return the caller to where they started, whether it finishes or fails.

Uses the real run() and make_sim_folders(), with the heavy steps (BNG2.pl,
basis-state sampling, writing files) replaced by stand-ins that record the
folder they ran in, so this needs no simulation stack.
"""
import os

import pytest

STEPS = ["copy_run_network", "write_static_files", "run_BNGL_on_file",
         "sample_basis_states", "write_dynamic_files", "make_analysis_notebook"]


def _stub_steps(wc, monkeypatch, fail_at=None):
    ran = []

    def make(name):
        def step(*args, **kwargs):
            ran.append((name, os.getcwd()))
            if name == fail_at:
                raise RuntimeError("simulated failure in " + name)
        return step

    for name in STEPS:
        monkeypatch.setattr(wc, name, make(name))
    return ran


def test_returns_to_starting_folder(bare_weconvert, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wc = bare_weconvert(fname="mysim", copy_run_net=False)
    ran = _stub_steps(wc, monkeypatch)
    wc.run()
    assert os.getcwd() == str(tmp_path)
    assert (tmp_path / "mysim" / "bngl_conf").is_dir()
    # the steps themselves still run inside the sim folder, as they expect
    assert [name for name, _ in ran] == STEPS[1:]
    assert all(cwd == str(tmp_path / "mysim") for _, cwd in ran)


def test_returns_to_starting_folder_after_a_failure(bare_weconvert, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wc = bare_weconvert(fname="mysim", copy_run_net=False)
    _stub_steps(wc, monkeypatch, fail_at="run_BNGL_on_file")
    with pytest.raises(RuntimeError, match="run_BNGL_on_file"):
        wc.run()
    assert os.getcwd() == str(tmp_path)
