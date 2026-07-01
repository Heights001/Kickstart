import engine


def test_package_importable() -> None:
    assert engine.__doc__ is not None
