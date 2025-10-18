import pytest

from pygobject_docs.inspect import patch_gi_overrides


@pytest.fixture(scope="session", autouse=True)
def overrides():
    patch_gi_overrides()
