"""Local configuration resolution only; no secrets files or services are read."""

import os
import unittest
from unittest.mock import patch

from configuration import resolve_api_key


class MissingSecrets:
    def get(self, name):
        raise FileNotFoundError("No local secrets file")


class ConfigurationTests(unittest.TestCase):
    def test_missing_streamlit_file_falls_back_to_environment(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test"}, clear=True):
            self.assertEqual(resolve_api_key("OPENAI_API_KEY", MissingSecrets()), "offline-test")

    def test_configured_streamlit_value_has_priority(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "environment-test"}, clear=True):
            self.assertEqual(resolve_api_key("OPENAI_API_KEY", {"OPENAI_API_KEY": "secrets-test"}), "secrets-test")

    def test_missing_configuration_returns_none(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(resolve_api_key("OPENAI_API_KEY", MissingSecrets()))

    def test_empty_secrets_value_uses_environment(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test"}, clear=True):
            self.assertEqual(resolve_api_key("OPENAI_API_KEY", {"OPENAI_API_KEY": ""}), "offline-test")


if __name__ == "__main__":
    unittest.main()
