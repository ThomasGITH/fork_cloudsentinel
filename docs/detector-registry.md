# Detector registry

Phase A introduces manifest discovery without changing the existing CGNN training or detection paths. A detector is discovered when a direct child of the `detectors` directory contains a `manifest.yaml` file.

The registry uses `yaml.safe_load`, limits each manifest to 1 MB, rejects paths that resolve outside the configured detector directory, validates required fields and parameter defaults, and rejects duplicate detector IDs. Discovery validates the `entry_point` string but deliberately does not import it.

Schema version 1 requires:

- `schema_version`
- `id`
- `name`
- `version`
- `description`
- `supported_modalities`
- `input_requirements`
- `training_parameters`
- `entry_point`

Each training parameter has `type`, `default`, and `description`. Nullable defaults must explicitly set `nullable: true`.

`GET /detectors` returns:

```json
{
  "detectors": [
    {
      "schema_version": 1,
      "id": "cgnn"
    }
  ]
}
```

The abbreviated object above is only illustrative; the endpoint returns the full validated manifest.

For local development, install the service requirements and shared package from the repository root:

```shell
python -m pip install -r learning_adaptation/requirements.txt
python -m pip install -e .
```

Build the learning-adaptation image from the repository root so the shared package is available in the build context:

```shell
docker build -f learning_adaptation/Dockerfile -t learning_adaptation .
```

Run the focused tests with:

```shell
python -m unittest tests.test_detectors
```
