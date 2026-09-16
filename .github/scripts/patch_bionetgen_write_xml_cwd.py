"""
Patches a second, separate instance of the same bug class already fixed
upstream in bionetgen/modelapi/model.py's setup_simulator(): a subprocess
is run with cwd=temp_folder (so BNG2.pl writes its output file *inside*
temp_folder), but the check/read of that output file afterward uses a
bare relative filename -- resolved against this process's own cwd, not
temp_folder -- so it never finds what the subprocess actually wrote.

This instance lives in bionetgen/modelapi/bngfile.py's write_xml(),
in both the "bngxml" branch (temp.xml) and the "sbml" branch
(temp_sbml.xml) -- the second is the one that broke webng's
libRoadRunner propagator with "SBML generation did not produce
temp_sbml.xml". Neither branch was touched by the existing upstream fix
in model.py, since that's a different function entirely.

Not yet reported upstream as of this writing -- remove this step once
RuleWorld/PyBioNetGen has its own fix merged and released.
"""
import os
import bionetgen

path = os.path.join(os.path.dirname(bionetgen.__file__), "modelapi", "bngfile.py")
print("patching:", path)

with open(path) as f:
    src = f.read()

replacements = [
    (
        'if not os.path.exists("temp.xml"):',
        'if not os.path.exists(os.path.join(temp_folder, "temp.xml")):',
    ),
    (
        'with open("temp.xml", "r", encoding="UTF-8") as f:',
        'with open(os.path.join(temp_folder, "temp.xml"), "r", encoding="UTF-8") as f:',
    ),
    (
        'if not os.path.exists("temp_sbml.xml"):',
        'if not os.path.exists(os.path.join(temp_folder, "temp_sbml.xml")):',
    ),
    (
        'with open("temp_sbml.xml", "r", encoding="UTF-8") as f:',
        'with open(os.path.join(temp_folder, "temp_sbml.xml"), "r", encoding="UTF-8") as f:',
    ),
]

missing = [old for old, _ in replacements if old not in src]
if missing:
    raise SystemExit(
        "patch_bionetgen_write_xml_cwd.py: expected text not found (upstream "
        "code may have changed) -- refusing to patch blindly. Missing:\n"
        + "\n".join(missing)
    )

for old, new in replacements:
    count = src.count(old)
    if count != 1:
        raise SystemExit(
            "patch_bionetgen_write_xml_cwd.py: expected exactly one occurrence "
            "of {!r}, found {}".format(old, count)
        )
    src = src.replace(old, new)

with open(path, "w") as f:
    f.write(src)

print("patched successfully")