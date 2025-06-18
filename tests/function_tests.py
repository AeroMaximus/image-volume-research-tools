import pytest

from training_data_selector import get_total_size, collect_file_paths

def test_get_total_size(tmp_path, capsys):
    # Create test files with known content
    file1 = tmp_path / "file1.txt"
    file1.write_text("abcd")  # 4 bytes

    file2 = tmp_path / "file2.txt"
    file2.write_text("12345678")  # 8 bytes

    # Create an invalid file path
    invalid_file = tmp_path / "does_not_exist.txt"

    # Case 1: Only valid files
    total = get_total_size([str(file1), str(file2)])
    assert total == 12

    # Case 2: One invalid file
    total_with_invalid = get_total_size([str(file1), str(invalid_file)])
    assert total_with_invalid == 4

    # Capture warning output
    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "does_not_exist.txt" in captured.out

    # Case 3: Empty list
    assert get_total_size([]) == 0

def test_collect_file_paths(tmp_path):
    # Create test files
    (tmp_path / "file1.txt").write_text("test1")
    (tmp_path / "file2.txt").write_text("test2")
    (tmp_path / "ignore.md").write_text("nope")

    # Test nested files
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    (nested_dir / "nested1.txt").write_text("nested_file1")
    (nested_dir / "nested2.txt").write_text("nested_file2")
    (nested_dir / "ignore.md").write_text("not a txt")

    # Test double nested
    double_nested_dir = nested_dir / "double_nested"
    double_nested_dir.mkdir()
    (double_nested_dir / "double_nested1.txt").write_text("double_nested_file1")
    (double_nested_dir / "double_nested2.txt").write_text("double_nested_file2")
    (double_nested_dir / "ignore.md").write_text("not a txt")

    # Collect .txt files
    txt_paths = collect_file_paths(tmp_path, accepted_extensions=('.txt',))

    expected_txt_paths = [
        str(tmp_path / "file1.txt"),
        str(tmp_path / "file2.txt"),
        str(nested_dir / "nested1.txt"),
        str(nested_dir / "nested2.txt"),
        str(double_nested_dir / "double_nested1.txt"),
        str(double_nested_dir / "double_nested2.txt")
    ]

    assert txt_paths == expected_txt_paths

    # Test the collection of multiple extensions
    all_paths = collect_file_paths(tmp_path, accepted_extensions=('.txt', '.md'))

    expected_all_paths = [
        str(tmp_path / "file1.txt"),
        str(tmp_path / "file2.txt"),
        str(tmp_path / "ignore.md"),
        str(nested_dir / "ignore.md"),
        str(nested_dir / "nested1.txt"),
        str(nested_dir / "nested2.txt"),
        str(double_nested_dir / "double_nested1.txt"),
        str(double_nested_dir / "double_nested2.txt"),
        str(double_nested_dir / "ignore.md")
    ]

    assert all_paths == expected_all_paths

    # Test directory without any accepted extensions
    with pytest.raises(ValueError):
        collect_file_paths(tmp_path, accepted_extensions=".wav")

    # Single valid file path
    path_pass = collect_file_paths(str(tmp_path / "file1.txt"), accepted_extensions=('.txt',))
    assert path_pass == [str(tmp_path / "file1.txt")]

    # Single invalid file path
    invalid_path = tmp_path / "file1.wav"
    invalid_path.write_text("dummy")
    with pytest.raises(ValueError):
        collect_file_paths(str(invalid_path), accepted_extensions=('.txt',))