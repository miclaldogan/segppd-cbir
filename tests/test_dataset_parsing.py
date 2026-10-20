import pytest
from segppd_cbir.labels import parse_filename, parse_fullname


def test_filename_ok():
    assert parse_filename("07016.png") == (70, 16)
    assert parse_filename("10109.png") == (101, 9)


@pytest.mark.parametrize("bad", ["7016.png", "070160.png", "abc12.png", "07016.jpg"])
def test_filename_ambiguous_fails(bad):
    with pytest.raises(ValueError):
        parse_filename(bad)


def test_fullname_crlf_and_no_trailing_newline(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"001 apple_healthy\r\n002 apple_powdery mildew")
    t = parse_fullname(p)
    assert t[1]["is_healthy_name"] and t[2]["crop"] == "apple" and t[2]["condition"] == "powdery mildew"


def test_fullname_duplicate_id_fails(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("001 a_x\n001 b_y\n")
    with pytest.raises(ValueError):
        parse_fullname(p)
