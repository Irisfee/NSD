#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions"""

from __future__ import annotations
from typing import Optional, Union
import os
from shlex import split
from subprocess import run


def conform_sub_id(sub_id: str, with_prefix=False) -> str:
    """Conform sub_id."""

    if not isinstance(sub_id, str):
        raise ValueError("Argument sub_id should be a string. (e.g., '001')")
    if (sub_id.startswith("sub-")) and (not with_prefix):
        sub_id = sub_id[4:]
    if (not sub_id.startswith("sub-")) and with_prefix:
        sub_id = f"sub-{sub_id}"

    return sub_id


def check_file_list(file_list: list[Union[os.PathLike, str]], n: Optional[int] = None):
    """Check the number of elements in a list of filenames."""

    if len(file_list) == 0:
        raise Exception("No file is found.")
    if n and len(file_list) != n:
        raise Exception("The number of returned files doesn't meet expectation.")


def _check_space(space: str, valid_list: list[str] = ["func1pt8mm", "func1mm"]):
    """Check input argument space."""
    assert space in valid_list, f"Valid space option: {valid_list}."


def _check_hemi(hemi: str, valid_list: list[str] = ["L", "R"]):
    """Check argument hemi."""
    assert hemi in valid_list, f"Valid hemi option: {valid_list}."


#####################
# Commandline related
#####################


def run_cmd(cmd: Union[str, list[str]], print_output: bool = True, shell: bool = False, **kwargs):

    if shell:
        res = run(cmd, shell=True, capture_output=True, encoding="utf-8", **kwargs)
    else:
        if isinstance(cmd, str):
            cmd = split(cmd)
        res = run(cmd, capture_output=True, encoding="utf-8", **kwargs)

    if print_output:
        if res.stdout != "":
            print(res.stdout.rstrip("\n"), flush=True)
        if res.stderr != "":
            print(res.stderr, flush=True)

    return res


######################
# Surface file related
######################


def sanitize_gii_metadata(
    in_file: os.PathLike,
    out_file: os.PathLike,
    gim_atr: dict = {},
    gim_meta: dict = {},
    da_atr: dict = {},
    da_meta: dict = {},
    clean_provenance: bool = False,
) -> os.PathLike:

    # Fix Gifti image metadata `Version`
    # When gifti file is processed by wb_command, the field `version`
    # in Gifti image metadata will be set to 1. This will cause error
    # when loading data to Freeview. Fix by setting it to 1.0.
    sanitize_cmd = [
        "gifti_tool",
        "-infile",
        in_file,
        "-write_gifti",
        out_file,
        "-mod_gim_atr",
        "Version",
        "1.0",
    ]

    # Replace user specified fields
    if gim_atr:
        for key, value in gim_atr.items():
            sanitize_cmd += ["-mod_gim_atr", key, value]
    if gim_meta:
        for key, value in gim_meta.items():
            sanitize_cmd += ["-mod_gim_meta", key, value]
    if da_atr:
        for key, value in da_atr.items():
            sanitize_cmd += ["-mod_DA_atr", key, value]
    if da_meta:
        for key, value in da_meta.items():
            sanitize_cmd += ["-mod_DA_meta", key, value]
    run_cmd(sanitize_cmd)

    # Cleanup provenance metadata added by wb_command
    if clean_provenance:
        wb_gim_meta = {
            "ProgramProvenance": "",
            "Provenance": "",
            "WorkingDirectory": "",
        }
        for key, value in wb_gim_meta.items():
            sanitize_cmd += ["-mod_gim_meta", key, value]

    # Verify output gifti file
    run_cmd(["gifti_tool", "-infile", out_file, "-gifti_test"])

    return out_file
