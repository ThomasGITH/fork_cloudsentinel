import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "external_plugin.sh"
ADMIN_MANIFEST = ROOT / "k8s" / "plugin-admin-pod.yml"
TUTORIAL = ROOT / "docs" / "external-plugin-tutorial.md"


class ExternalPluginOperatorUXTests(unittest.TestCase):
    def run_script(self, *arguments: str, environment=None):
        return subprocess.run(
            [str(SCRIPT), *arguments],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def make_source(self, root: Path) -> Path:
        source = root / "operator_fixture"
        source.mkdir()
        (source / "manifest.yaml").write_text("id: operator-fixture\n", encoding="utf-8")
        (source / "adapter.py").write_text("class Adapter: pass\n", encoding="utf-8")
        (source / "requirements.lock").write_text("PyYAML==6.0.2\n", encoding="utf-8")
        return source

    def make_fake_environment(self, root: Path):
        executable_root = root / "bin"
        executable_root.mkdir()
        log = root / "kubectl.log"
        fake = executable_root / "kubectl"
        fake.write_text(
            "#!/usr/bin/env bash\n"
            "set -eu\n"
            "printf '%s\\n' \"$*\" >> \"$FAKE_KUBECTL_LOG\"\n"
            "case \" $* \" in\n"
            "  *\" get pod \"*\" -o jsonpath=\"*) printf 'plugin-admin' ;;\n"
            "  *\" cloudsentinel-plugin validate \"*)\n"
            "    printf '{\"status\":\"valid\",\"detector_id\":\"%s\",\"detector_version\":\"%s\"}\\n' \"$FAKE_PLUGIN_ID\" \"$FAKE_PLUGIN_VERSION\" ;;\n"
            "  *\" cloudsentinel-plugin install \"*)\n"
            "    printf '{\"detector_id\":\"%s\",\"detector_version\":\"%s\"}\\n' \"$FAKE_PLUGIN_ID\" \"$FAKE_PLUGIN_VERSION\" ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{executable_root}{os.pathsep}{environment['PATH']}",
                "FAKE_KUBECTL_LOG": str(log),
                "FAKE_PLUGIN_ID": "operator-fixture",
                "FAKE_PLUGIN_VERSION": "1.0.0",
            }
        )
        return environment, log

    def test_script_has_valid_bash_syntax(self):
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT)], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_publish_rejects_relative_and_incomplete_sources_before_kubectl(self):
        relative = self.run_script(
            "publish", "relative/plugin", "--id", "safe-detector", "--version", "1.0.0"
        )
        self.assertNotEqual(relative.returncode, 0)
        self.assertIn("absolute path", relative.stderr)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "incomplete"
            source.mkdir()
            incomplete = self.run_script(
                "publish", str(source), "--id", "safe-detector", "--version", "1.0.0"
            )
        self.assertNotEqual(incomplete.returncode, 0)
        self.assertIn("manifest.yaml", incomplete.stderr)

    def test_publish_validates_installs_activates_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            environment, log = self.make_fake_environment(root)
            result = self.run_script(
                "publish",
                str(source),
                "--id",
                "operator-fixture",
                "--version",
                "1.0.0",
                "--activate",
                environment=environment,
            )
            calls = log.read_text(encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cloudsentinel-plugin validate", calls)
        self.assertIn("cloudsentinel-plugin install", calls)
        self.assertIn("cloudsentinel-plugin activate operator-fixture 1.0.0", calls)
        self.assertIn("delete pod -n cloudsentinel plugin-admin", calls)
        self.assertIn("Published and activated", result.stdout)

    def test_publish_without_activate_does_not_switch_active_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            environment, log = self.make_fake_environment(root)
            result = self.run_script(
                "publish",
                str(source),
                "--id",
                "operator-fixture",
                "--version",
                "1.0.0",
                environment=environment,
            )
            calls = log.read_text(encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("cloudsentinel-plugin activate", calls)
        self.assertIn("without changing the active release", result.stdout)

    def test_manifest_identity_mismatch_fails_and_cleanup_still_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            environment, log = self.make_fake_environment(root)
            environment["FAKE_PLUGIN_ID"] = "different-detector"
            result = self.run_script(
                "publish",
                str(source),
                "--id",
                "operator-fixture",
                "--version",
                "1.0.0",
                environment=environment,
            )
            calls = log.read_text(encoding="utf-8")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match --id", result.stderr)
        self.assertNotIn("cloudsentinel-plugin install", calls)
        self.assertIn("delete pod -n cloudsentinel plugin-admin", calls)

    def test_rollback_uses_existing_cli_and_keep_option_preserves_pod(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment, log = self.make_fake_environment(root)
            result = self.run_script(
                "rollback",
                "operator-fixture",
                "1.0.0",
                "--keep-admin-pod",
                environment=environment,
            )
            calls = log.read_text(encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cloudsentinel-plugin rollback operator-fixture 1.0.0", calls)
        self.assertNotIn("delete pod", calls)

    def test_script_and_tutorial_reference_real_generic_resources(self):
        script = SCRIPT.read_text(encoding="utf-8")
        tutorial = TUTORIAL.read_text(encoding="utf-8")
        self.assertTrue(SCRIPT.stat().st_mode & 0o111)
        self.assertTrue(ADMIN_MANIFEST.is_file())
        self.assertIn("trap cleanup EXIT", script)
        self.assertIn("k8s/plugin-admin-pod.yml", script)
        self.assertNotIn("local-outlier-factor", script)
        self.assertIn("./scripts/external_plugin.sh publish", tutorial)
        self.assertIn("./scripts/external_plugin.sh rollback", tutorial)
        self.assertIn("k8s/plugin-admin-pod.yml", ADMIN_MANIFEST.as_posix())


if __name__ == "__main__":
    unittest.main()
