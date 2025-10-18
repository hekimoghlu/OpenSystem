from pygobject_docs.generate import order


def test_order():
    libs = ["A-1.0", "B-1.0", "C-1.0", "D-1.0"]
    top = ["D", "A", "C", "B"]

    assert order(libs, top) == ["D-1.0", "A-1.0", "C-1.0", "B-1.0"]
