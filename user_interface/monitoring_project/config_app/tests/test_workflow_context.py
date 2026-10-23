from django.test import SimpleTestCase

from config_app.workflow_context import (
    decode_context,
    encode_context,
    new_context,
    progress_context,
    update_context,
)


class WorkflowContextTests(SimpleTestCase):
    def test_signed_context_survives_component_handoffs(self):
        context = new_context(
            dataset_id="ds_demo",
            dataset_version=3,
            dataset_name="Checkout metrics",
        )
        context = update_context(context, training_run_id="run-1")
        context = update_context(context, comparison_id="cmp-1", model_id="model-2")

        restored = decode_context(encode_context(context))

        self.assertEqual(restored["dataset_id"], "ds_demo")
        self.assertEqual(restored["training_run_id"], "run-1")
        self.assertEqual(restored["comparison_id"], "cmp-1")
        self.assertEqual(restored["model_id"], "model-2")

    def test_tampered_context_is_rejected(self):
        token = encode_context(new_context(dataset_id="ds_demo"))
        self.assertIsNone(decode_context(token + "changed"))

    def test_progress_marks_prior_steps_complete(self):
        progress = progress_context(new_context(dataset_id="ds_demo"), "comparison")
        states = {item["key"]: item["state"] for item in progress["steps"]}
        self.assertEqual(states["dataset"], "complete")
        self.assertEqual(states["training"], "complete")
        self.assertEqual(states["comparison"], "current")
        self.assertEqual(states["monitoring"], "upcoming")
