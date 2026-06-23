import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from apps.api.app import PROJECT_ROOT, app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_image_endpoint_is_restricted_to_data_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.png"
            Image.new("RGB", (8, 8)).save(image_path)
            with patch.dict(os.environ, {"MOONDREAM_DATA_ROOT": tmp}):
                allowed = self.client.get("/api/image", params={"path": str(image_path)})
                blocked = self.client.get("/api/image", params={"path": str(PROJECT_ROOT / "README.md")})
            self.assertEqual(allowed.status_code, 200)
            self.assertEqual(blocked.status_code, 400)


if __name__ == "__main__":
    unittest.main()
