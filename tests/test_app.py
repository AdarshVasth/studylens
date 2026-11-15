import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from app.core import Library, chunk_page
from app import main


class StudyLensTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.lib = Library(self.temp.name)
        self.swap = patch.object(main, 'library', self.lib)
        self.swap.start()
        self.env = patch.dict(os.environ, {'STUDYLENS_SEMANTIC': '0'})
        self.env.start()
        self.client = TestClient(main.app)

    def tearDown(self):
        self.swap.stop()
        self.env.stop()
        self.temp.cleanup()

    def test_chunks_overlap_and_cover_tail(self):
        parts = chunk_page(' '.join(f'word{i}' for i in range(300)))
        self.assertIn('word299', parts[-1])
        self.assertEqual(parts[0].split()[-25:], parts[1].split()[:25])

    def test_upload_search_and_page_provenance(self):
        response = self.client.post('/api/documents', files={'file': ('notes.txt', b'Paging maps virtual pages to physical frames. A page fault loads a missing page.')})
        self.assertEqual(response.status_code, 201)
        result = self.client.post('/api/ask', json={'question': 'What is paging?'}).json()
        self.assertEqual(result['sources'][0]['page'], 1)
        self.assertIn('Paging', result['answer'])
        self.assertFalse(result['generated'])

    def test_persistence_and_delete(self):
        doc = self.lib.add('notes', ['Context switching saves registers and restores process state.'])
        self.assertEqual(Library(self.temp.name).list()[0]['id'], doc['id'])
        self.assertEqual(self.client.delete('/api/documents/' + doc['id']).status_code, 200)
        self.assertEqual(self.lib.chunks(), [])
        self.assertEqual(self.client.delete('/api/documents/' + doc['id']).status_code, 404)

    def test_bad_uploads_and_size(self):
        for name, data in [('bad.exe', b'test'), ('broken.pdf', b'not a pdf'), ('empty.txt', b'')]:
            self.assertEqual(self.client.post('/api/documents', files={'file': (name, data)}).status_code, 400)
        self.assertEqual(self.client.post('/api/documents', files={'file': ('large.txt', b'x' * (8 * 1024 * 1024 + 1))}).status_code, 413)

    def test_no_results_and_blank_input(self):
        self.assertEqual(self.client.post('/api/ask', json={'question': 'What is paging?'}).json()['sources'], [])
        self.assertEqual(self.client.post('/api/ask', json={'question': '   '}).status_code, 422)
        self.client.post('/api/demo')
        self.assertEqual(self.client.post('/api/ask', json={'question': 'photosynthesis chlorophyll'}).json()['sources'], [])

    def test_demo_idempotence_and_ranking(self):
        self.client.post('/api/demo')
        self.client.post('/api/demo')
        self.assertEqual(len(self.lib.list()), 1)
        sources, _ = self.lib.search('four necessary deadlock conditions')
        self.assertEqual(sources[0]['page'], 3)

    def test_valid_generated_answer(self):
        self.client.post('/api/demo')
        response = Mock()
        response.json.return_value = {'message': {'content': '{"answer":"Context switching saves CPU state [S1].","citations":["S1"]}'}}
        with patch('app.main.httpx.post', return_value=response):
            result = self.client.post('/api/ask', json={'question': 'context switch', 'generate': True}).json()
        self.assertTrue(result['generated'])

    def test_model_failure_and_invented_citation_fallback(self):
        self.client.post('/api/demo')
        response = Mock()
        response.json.return_value = {'message': {'content': '{"answer":"Made up [S99]", "citations":["S99"]}'}}
        for settings in [{'return_value': response}, {'side_effect': main.httpx.ConnectError('offline')}]:
            with patch('app.main.httpx.post', **settings):
                result = self.client.post('/api/ask', json={'question': 'context switch', 'generate': True}).json()
            self.assertFalse(result['generated'])
            self.assertIn('warning', result)

    def test_pdf_page_numbers(self):
        self.lib.add('lecture.pdf', ['Scheduling gives processes CPU time.', 'Virtual memory paging translates pages into physical frames.'])
        self.assertEqual(self.lib.search('paging')[0][0]['page'], 2)

    def test_hybrid_embeddings_fused_and_invalidated(self):
        self.lib.add('notes.txt', ['Processes save registers during a context switch.'])
        import numpy as np
        encoder = Mock()
        encoder.encode_document.side_effect = lambda texts, **kw: np.array([[1., 0.]] * len(texts))
        encoder.encode_query.return_value = np.array([[1., 0.]])
        self.lib.model = encoder
        with patch.dict(os.environ, {'STUDYLENS_SEMANTIC': '1'}):
            sources, mode = self.lib.search('CPU handover')
            self.assertEqual(mode, 'hybrid')
            self.assertTrue(sources)
            self.lib.search('CPU handover')
            self.assertEqual(encoder.encode_document.call_count, 1)
            self.lib.add('more.txt', ['Threads share memory within a process.'])
            self.lib.search('CPU handover')
            self.assertEqual(encoder.encode_document.call_count, 2)


if __name__ == '__main__':
    unittest.main()
