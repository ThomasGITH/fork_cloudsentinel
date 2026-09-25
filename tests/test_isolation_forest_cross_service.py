import io
import json
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np
import requests
from werkzeug.serving import make_server

from anomaly_detection.isolation_forest.app import create_app
from detectors import get_adapter
from learning_adaptation.isolation_forest_promotion import (
    promote_isolation_forest_model,
)


class IsolationForestCrossServiceTests(unittest.TestCase):
    def test_real_training_http_promotion_and_prediction_roundtrip(self):
        rng = np.random.default_rng(42)
        train = rng.normal(0.0, 0.25, size=(80, 2))
        normal_test = rng.normal(0.0, 0.25, size=(20, 2))
        anomalous_test = rng.normal(8.0, 0.1, size=(5, 2))
        test = np.vstack((normal_test, anomalous_test))
        labels = np.concatenate((np.zeros(20), np.ones(5)))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_artifacts = root / "training" / "roundtrip-model"
            model_storage = root / "detection-models"
            adapter = get_adapter("isolation-forest")
            adapter.train(
                train,
                training_artifacts,
                training_parameters={"n_estimators": 25, "random_state": 42},
                model_id="roundtrip-model",
            )
            adapter.evaluate(test, labels, training_artifacts)

            app = create_app(
                {
                    "TESTING": True,
                    "IF_MODEL_STORAGE_ROOT": str(model_storage),
                    "IF_RESULTS_STORAGE_ROOT": str(root / "detection-results"),
                }
            )
            server = make_server("127.0.0.1", 0, app, threaded=True)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            service_url = f"http://127.0.0.1:{server.server_port}"
            try:
                promotion = promote_isolation_forest_model(
                    training_artifacts,
                    "roundtrip-model",
                    service_url=service_url,
                    timeout=(2.0, 10.0),
                )

                installed = (
                    model_storage / "isolation-forest" / "roundtrip-model"
                )
                self.assertEqual(
                    sorted(path.name for path in installed.iterdir()),
                    [
                        "model.joblib",
                        "model_evaluation.json",
                        "model_metadata.json",
                    ],
                )
                self.assertEqual(promotion["status"], "available")

                csv_buffer = io.StringIO()
                np.savetxt(csv_buffer, test, delimiter=",")
                response = requests.post(
                    f"{service_url}/detect_anomalies",
                    files={
                        "test_array": (
                            "test.csv",
                            io.BytesIO(csv_buffer.getvalue().encode("utf-8")),
                            "text/csv",
                        )
                    },
                    data={
                        "test_info": json.dumps(
                            {
                                "data": {
                                    "detector_id": "isolation-forest",
                                    "model": "roundtrip-model",
                                    "iteration": "0",
                                }
                            }
                        )
                    },
                    timeout=(2.0, 10.0),
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual(result["status"], "success")
                self.assertEqual(result["detector_id"], "isolation-forest")
                self.assertEqual(result["model_id"], "roundtrip-model")
                self.assertEqual(result["iteration"], "0")
                self.assertGreaterEqual(result["percentage"], 0.0)
                self.assertLessEqual(result["percentage"], 100.0)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5.0)


if __name__ == "__main__":
    unittest.main()
