from datetime import datetime, timedelta, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = ROOT / "user_interface/monitoring_project/config_app"
SPEC = spec_from_file_location("ui_formatting", UI_ROOT / "ui_formatting.py")
formatting = module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(formatting)


class UiReadableModelsAndDatesTests(unittest.TestCase):
    def test_iso_instants_are_displayed_in_explicit_utc(self):
        self.assertEqual(
            formatting.format_ui_datetime("2026-10-10T17:20:00Z"),
            "10 Oct 2026, 17:20 UTC",
        )
        self.assertEqual(
            formatting.format_ui_datetime("2026-10-10T19:20:00+02:00"),
            "10 Oct 2026, 17:20 UTC",
        )
        self.assertEqual(
            formatting.format_ui_datetime(
                datetime(2026, 10, 10, 18, 20, tzinfo=timezone(timedelta(hours=1)))
            ),
            "10 Oct 2026, 17:20 UTC",
        )

    def test_missing_and_unparseable_values_are_honest(self):
        self.assertEqual(formatting.format_ui_datetime(None), "—")
        self.assertEqual(formatting.format_ui_datetime(None, "Unavailable"), "Unavailable")
        self.assertEqual(formatting.format_ui_datetime("not-a-date"), "not-a-date")

    def test_models_and_training_use_readable_names_and_shared_date_filter(self):
        overview = (UI_ROOT / "templates/config_app/models/overview.html").read_text()
        detail = (UI_ROOT / "templates/config_app/models/model_detail.html").read_text()
        run = (UI_ROOT / "templates/config_app/models/run_detail.html").read_text()
        view = (UI_ROOT / "views/models_ui.py").read_text()
        self.assertIn("{{ model.display_name }}", overview)
        self.assertIn("{{ child.model_display_name }}", overview)
        self.assertIn("<h1>{{ model.display_name }}</h1>", detail)
        self.assertIn("_add_model_display_names", view)
        for template in (overview, detail, run):
            self.assertIn("ui_datetime", template)

    def test_dynamic_pages_use_one_date_formatter(self):
        base = (UI_ROOT / "templates/base.html").read_text()
        polling = (UI_ROOT / "static/js/models_run.js").read_text()
        monitoring = (UI_ROOT / "static/js/monitoring_history.js").read_text()
        catalogue = (UI_ROOT / "static/js/data_catalogue_detail.js").read_text()
        comparison = (UI_ROOT / "static/js/comparison_detail.js").read_text()
        helper = (UI_ROOT / "static/js/ui_datetime.js").read_text()
        self.assertIn("js/ui_datetime.js", base)
        self.assertIn("CloudSentinel.formatDateTime", polling)
        self.assertIn("CloudSentinel.formatDateTime", monitoring)
        self.assertIn("CloudSentinel.formatDateTime", catalogue)
        self.assertIn("CloudSentinel.formatDateTime", comparison)
        self.assertIn('timeZone: "UTC"', helper)
        self.assertNotIn("toLocaleString", monitoring)

    def test_machine_datetime_inputs_keep_iso_values(self):
        source_fields = (UI_ROOT / "templates/config_app/data_catalogue/_source_fields.html").read_text()
        self.assertIn('type="datetime-local"', source_fields)
        self.assertIn('data-initial="{{ source.start_time|default:\'\' }}"', source_fields)
        self.assertNotIn("ui_datetime", source_fields)

    def test_robustness_explains_labelled_context_filter(self):
        template = (UI_ROOT / "templates/config_app/comparison/_robustness.html").read_text()
        self.assertIn("Labelled contexts only", template)
        self.assertIn("stored ground-truth labels", template)
        self.assertIn("Precision, recall and F1", template)


if __name__ == "__main__":
    unittest.main()
