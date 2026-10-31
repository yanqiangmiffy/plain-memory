import threading
import time

import pytest

from plain_memory import retrieval
from plain_memory.retrieval import IndexCache


class FakeIndex:
    def __init__(self, rows):
        self.rows = rows


class FakeStore:
    def __init__(self, sizes, delay=0.0):
        self.sizes = dict(sizes)
        self.delay = delay
        self.builds = []

    def fingerprint(self, user_id):
        return self.sizes.get(user_id, 0), self.sizes.get(user_id, 0)

    def snapshot(self, user_id):
        self.builds.append(user_id)
        time.sleep(self.delay)
        return [{"user": user_id}] * self.sizes[user_id]


@pytest.fixture(autouse=True)
def fake_index(monkeypatch):
    monkeypatch.setattr(retrieval, "UserIndex", FakeIndex)


def test_unchanged_user_is_served_from_memory_and_a_changed_one_rebuilds():
    store = FakeStore({"a": 3})
    cache = IndexCache(capacity=4, max_units=100)
    first = cache.get(store, "a")
    assert cache.get(store, "a") is first
    store.sizes["a"] = 4  # a new Add changes the fingerprint
    assert cache.get(store, "a") is not first
    assert store.builds == ["a", "a"]
    assert cache.get(store, "nobody") is None


def test_total_units_are_bounded_and_the_newest_user_always_stays():
    store = FakeStore({"a": 6, "b": 6, "c": 30})
    cache = IndexCache(capacity=10, max_units=10)
    cache.get(store, "a")
    cache.get(store, "b")
    assert list(cache.items) == ["b"]  # a + b would be 12 units
    cache.get(store, "c")  # larger than the whole budget: still kept alone
    assert list(cache.items) == ["c"]


def test_user_count_is_bounded_least_recently_used_first():
    store = FakeStore({name: 1 for name in "abc"})
    cache = IndexCache(capacity=2, max_units=100)
    cache.get(store, "a")
    cache.get(store, "b")
    cache.get(store, "a")  # a is now the freshest
    cache.get(store, "c")
    assert list(cache.items) == ["a", "c"]


def test_concurrent_searches_for_one_user_build_the_index_once():
    store = FakeStore({"a": 2}, delay=0.2)
    cache = IndexCache()
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(cache.get(store, "a"))) for _ in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.builds == ["a"]
    assert len({id(r) for r in results}) == 1


def test_a_slow_rebuild_does_not_block_other_users():
    store = FakeStore({"slow": 2, "fast": 2}, delay=0.0)
    cache = IndexCache()
    cache.get(store, "fast")
    slow = "slow"
    gate = threading.Event()
    original = store.snapshot

    def slow_snapshot(user_id):
        if user_id == slow:
            gate.wait(5)
        return original(user_id)

    store.snapshot = slow_snapshot
    worker = threading.Thread(target=lambda: cache.get(store, slow))
    worker.start()
    time.sleep(0.1)
    started = time.monotonic()
    assert cache.get(store, "fast") is not None  # cached hit takes only the short global lock
    assert time.monotonic() - started < 1
    gate.set()
    worker.join()
