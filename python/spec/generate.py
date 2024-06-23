#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import json
import os
from pathlib import Path

import pkg_resources
from jinja2 import Environment, FileSystemLoader, select_autoescape

base = Path(os.path.dirname(os.path.abspath(__file__)))

env = Environment(
    loader=FileSystemLoader(base / "templates"),
    autoescape=select_autoescape(["cpp"]),
)

with open(
    pkg_resources.resource_filename("celerite2", "definitions.json"), "r"
) as f:
    data = json.load(f)

for n in range(len(data)):
    data[n]["rev_inputs"] = copy.deepcopy(
        data[n]["inputs"]
        + data[n]["outputs"]
        + data[n]["extra_outputs"]
        + [dict(arg, name="b" + arg["name"]) for arg in data[n]["outputs"]]
    )
    data[n]["rev_outputs"] = [
        dict(arg, name="b" + arg["name"]) for arg in data[n]["inputs"]
    ]

    for val in data[n]["inputs"]:
        val["is_output"] = False
    for val in data[n]["outputs"]:
        val["is_output"] = True
    for val in data[n]["extra_outputs"]:
        val["is_output"] = True
    for val in data[n]["rev_inputs"]:
        val["is_output"] = False
    for val in data[n]["rev_outputs"]:
        val["is_output"] = True

for name in ["driver.cpp", "backprop.cpp", "jax/xla_ops.cpp"]:
    template = env.get_template(name)
    result = template.render(spec=data)
    with open(base.parent / "celerite2" / name, "w") as f:
        f.write("// NOTE: This file was autogenerated\n")
        f.write("// NOTE: Changes should be made to the template\n\n")
        f.write(result)
