# External detector template

This directory is a deliberately incomplete starting point. It is outside the
built-in `detectors/` package and is never discovered automatically.

## Start a detector

Copy the directory to a new absolute working path:

```bash
cp -R examples/external_plugins/template_detector /home/ubuntu/plugins/my_detector
```

Then:

1. replace every `<replace-...>` value in `manifest.yaml`;
2. declare only parameters accepted by your implementation;
3. implement `validate_training()` and `run_training()` in `adapter.py`;
4. add implementation modules as needed using package-relative imports;
5. pin only already-installed runtime dependencies in `requirements.lock`;
6. validate and publish through `scripts/external_plugin.sh`.

The template intentionally fails manifest validation until its placeholders are
replaced. Its adapter intentionally rejects training until both method bodies
are implemented.

## Minimum package

The normal publication wrapper requires:

```text
manifest.yaml
adapter.py
requirements.lock
```

Additional regular files and Python modules are optional. `README.md` is
optional. Symlinks, virtual environments, `.git`, bytecode, sockets and other
special files are rejected.

See `docs/external-plugin-tutorial.md` for the complete field and publication
contract.
