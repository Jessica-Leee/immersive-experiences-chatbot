"""Ingestion contracts with external SDKs replaced at the import boundary."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[1] / "embedding_generator.py"


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.client.list_indexes.return_value.names.return_value = ["blckunicrn"]
        self.pinecone = Mock(return_value=self.client)
        self.specification = Mock()
        self.dotenv = Mock()
        self.pdf_open = Mock()
        # A tiny reversible token vocabulary isolates chunk boundaries from the
        # tokenizer package. It does not generate embeddings or contact a service.
        encoding = SimpleNamespace(encode=lambda text: list(text), decode=lambda tokens: "".join(tokens))
        self.modules = {
            "dotenv": SimpleNamespace(load_dotenv=self.dotenv),
            "fitz": SimpleNamespace(open=self.pdf_open),
            "tiktoken": SimpleNamespace(get_encoding=Mock(return_value=encoding)),
            "openai": SimpleNamespace(embeddings=Mock()),
            "pinecone": SimpleNamespace(Pinecone=self.pinecone, ServerlessSpec=self.specification),
        }
        with patch.dict("sys.modules", self.modules):
            spec = importlib.util.spec_from_file_location("ingestion_under_test", SOURCE)
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)

    def test_import_does_not_load_env_open_pdfs_or_construct_clients(self):
        self.dotenv.assert_not_called()
        self.pdf_open.assert_not_called()
        self.pinecone.assert_not_called()
        self.modules["openai"].embeddings.create.assert_not_called()

    def test_missing_api_configuration_fails_before_service_use(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "API key"):
                self.module.Generator()
            with self.assertRaisesRegex(ValueError, "PINECONE_API_KEY"):
                self.module.PineconeStore()
        self.pinecone.assert_not_called()
        self.modules["openai"].embeddings.create.assert_not_called()

    def test_chunk_boundaries_preserve_order_and_remainder(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test"}, clear=True):
            generator = self.module.Generator()
        self.assertEqual(generator.chunk_text_by_tokens("abcdefg", 3), ["abc", "def", "g"])
        self.assertEqual(generator.chunk_text_by_tokens("", 3), [])
        with self.assertRaises(ValueError):
            generator.chunk_text_by_tokens("abc", 0)
        with self.assertRaises(ValueError):
            generator.chunk_text_by_tokens("abc", -1)
        self.modules["openai"].embeddings.create.assert_not_called()

    def test_existing_index_is_reused_with_v3_sdk_interface(self):
        with patch.dict(os.environ, {"PINECONE_API_KEY": "offline-test"}, clear=True):
            self.module.PineconeStore()
        self.client.list_indexes.assert_called_once_with()
        self.client.create_index.assert_not_called()

    def test_absent_index_uses_embedding_dimensions_and_serverless_spec(self):
        self.client.list_indexes.return_value.names.return_value = []
        with patch.dict(os.environ, {"PINECONE_API_KEY": "offline-test"}, clear=True):
            self.module.PineconeStore()
        self.specification.assert_called_once_with(cloud="aws", region="us-east-1")
        self.client.create_index.assert_called_once_with(
            name="blckunicrn", dimension=1536, metric="cosine", spec=self.specification.return_value,
        )

    def test_mismatched_chunks_are_rejected_before_index_access(self):
        store = object.__new__(self.module.PineconeStore)
        store.pc = self.client
        with self.assertRaisesRegex(ValueError, "exactly one"):
            store.save_vectors([[0.1]], {"id": "synthetic", "source": "sample.pdf"}, [])
        self.client.Index.assert_not_called()

    def test_vector_records_keep_stable_ids_and_source_text(self):
        store = object.__new__(self.module.PineconeStore)
        store.pc = self.client
        store.index_name = "blckunicrn"
        store.save_vectors([[0.1], [0.2]], {"id": "synthetic", "source": "sample.pdf"}, ["first", "second"])
        vectors = [call.kwargs["vectors"][0] for call in self.client.Index.return_value.upsert.call_args_list]
        self.assertEqual([record[0] for record in vectors], ["synthetic_chunk_0", "synthetic_chunk_1"])
        self.assertEqual([record[2]["text"] for record in vectors], ["first", "second"])
        self.assertTrue(all(record[2]["source"] == "sample.pdf" for record in vectors))


if __name__ == "__main__":
    unittest.main()
