from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "user_interface/monitoring_project/config_app/static/js/monitoring_history.js"


def function_source(source, name, next_name):
    start = source.index(f"function {name}")
    return source[start : source.index(f"function {next_name}", start)]


class MonitoringPollingUiRegressionTests(unittest.TestCase):
    def setUp(self):
        self.source = SCRIPT.read_text()

    def test_unchanged_poll_payloads_do_not_rebuild_the_dom(self):
        sessions = function_source(self.source, "renderSessions", "metric")
        detail = function_source(self.source, "renderDetail", "schedulePoll")
        self.assertIn("state.sessionsRenderKey === key", sessions)
        self.assertIn("return false", sessions)
        self.assertIn("state.detailRenderKey === key", detail)
        self.assertIn("return false", detail)

    def test_chart_instance_is_reused_and_updated_without_animation(self):
        chart = function_source(self.source, "renderChart", "renderRcaDetail")
        self.assertIn("state.chart.canvas === canvas", chart)
        self.assertIn('state.chart.update("none")', chart)
        self.assertIn("state.chart.destroy()", chart)
        self.assertLess(
            chart.index("state.chart.canvas === canvas"),
            chart.index("state.chart.destroy()"),
        )
        self.assertIn("preservedChartCard", self.source)

    def test_terminal_rca_result_is_cached_without_unbounded_growth(self):
        loader = function_source(self.source, "loadRcaDetail", "renderRca")
        cache = function_source(self.source, "cacheRcaResult", "loadRcaDetail")
        self.assertIn("state.rcaResultCache.get(cacheKey)", loader)
        self.assertIn('["completed","failed","skipped"]', loader)
        self.assertIn("state.rcaResultCache.size > 50", cache)

    def test_background_tab_pauses_polling_and_foreground_resumes(self):
        self.assertIn('document.addEventListener("visibilitychange"', self.source)
        self.assertIn("if (document.hidden) clearTimeout(state.timer)", self.source)
        schedule = function_source(self.source, "schedulePoll", "loadAll")
        self.assertIn("state.shouldPoll && !document.hidden", schedule)

    def test_poll_errors_do_not_clear_the_existing_dashboard(self):
        loader = self.source[
            self.source.index("async function loadAll") :
            self.source.index("newer.addEventListener")
        ]
        catch = loader[loader.index("catch (error)") :]
        self.assertIn("showError(error.message)", catch)
        self.assertNotIn("detailNode.replaceChildren", catch)
        self.assertNotIn("listNode.replaceChildren", catch)


class MonitoringPollingDeploymentTests(unittest.TestCase):
    def test_ui_manifest_uses_image_containing_the_polling_fix(self):
        documents = list(
            yaml.safe_load_all(
                (ROOT / "k8s/monitoring_project-deployment.yml").read_text()
            )
        )
        deployment = next(item for item in documents if item and item.get("kind") == "Deployment")
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(
            container["image"],
            "jojojochem/monitoring_project:expansions-p3",
        )
        self.assertEqual(container["imagePullPolicy"], "IfNotPresent")


if __name__ == "__main__":
    unittest.main()
