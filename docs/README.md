# where is the users manual?

the users manual is at
[`documentation/pestpp_users_manual.md`](../documentation/pestpp_users_manual.md).
that is the file to read or edit.

the rendered version is at <https://pestpp.readthedocs.io/en/latest/users_guide.html>.

this folder is the source for the readthedocs site. `users_guide.md` here only
includes `users_manual_processed.md`, which is not in the repo - readthedocs makes it
during the build by running `preprocess_manual.py` on the manual (see
`.readthedocs.yaml`).
