"""Small reproducible retrieval smoke benchmark; not a general quality claim."""
import tempfile
from pathlib import Path
from app.core import Library

CASES = [('four necessary deadlock conditions', 3), ('context switch saves registers', 2),
         ('page fault virtual memory', 4), ('Round Robin time quantum', 1),
         ('waiting time turnaround arrival', 5)]


def main():
    with tempfile.TemporaryDirectory() as directory:
        library = Library(directory)
        library.add('OS notes', (Path(__file__).resolve().parent.parent / 'data/demo.txt').read_text().split('\n---\n'))
        hits = sum(bool(sources) and sources[0]['page'] == page
                   for query, page in CASES for sources, _ in [library.search(query)])
        print(f'Demo retrieval hit@1: {hits}/{len(CASES)} ({hits / len(CASES):.0%})')
        if hits != len(CASES):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
