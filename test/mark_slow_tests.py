#!/usr/bin/env python3
"""
Parse pytest XML report and mark tests that take longer than 20 seconds as slow.
"""

import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def parse_xml_report(xml_file):
    """Parse the pytest XML report and find slow tests."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    slow_tests = []

    for testsuite in root.findall("testsuite"):
        for testcase in testsuite.findall("testcase"):
            classname = testcase.get("classname", "")
            name = testcase.get("name", "")
            time_str = testcase.get("time", "0")

            try:
                time_seconds = float(time_str)
            except (ValueError, TypeError):
                time_seconds = 0.0

            if time_seconds > 20:
                # Extract file path from classname
                # Format is like: test.unit.test_file or test.unit.test_file.TestClass
                parts = classname.split(".")
                if len(parts) >= 3:
                    # test.unit.test_file -> test/unit/test_file.py
                    file_path = "/".join(parts) + ".py"

                    slow_tests.append(
                        {
                            "classname": classname,
                            "name": name,
                            "time": time_seconds,
                            "file_path": file_path,
                        }
                    )

    return sorted(slow_tests, key=lambda x: x["time"], reverse=True)


def group_tests_by_file(slow_tests):
    """Group tests by file and class."""
    by_file = defaultdict(lambda: defaultdict(list))

    for test in slow_tests:
        parts = test["classname"].split(".")
        file_key = ".".join(parts[:3])  # test.unit.test_file

        if len(parts) > 3:
            # Has a class name
            class_name = ".".join(parts[3:])
            by_file[file_key][class_name].append(test)
        else:
            # Module-level test
            by_file[file_key]["_module"].append(test)

    return by_file


def main():
    xml_file = Path("pytest_report.xml")

    if not xml_file.exists():
        print(f"XML report not found: {xml_file}")
        return

    print("Parsing pytest XML report...")
    slow_tests = parse_xml_report(xml_file)

    if not slow_tests:
        print("\nNo tests taking more than 20 seconds found!")
        return

    print(f"\nFound {len(slow_tests)} tests taking more than 20 seconds:\n")
    print("=" * 100)

    by_file = group_tests_by_file(slow_tests)

    total_time = sum(t["time"] for t in slow_tests)

    for file_key in sorted(by_file.keys()):
        file_path = file_key.replace(".", "/") + ".py"
        print(f"\n{file_path}")
        print("-" * 100)

        file_tests = by_file[file_key]
        file_time = 0

        for class_name in sorted(file_tests.keys()):
            tests = file_tests[class_name]
            class_time = sum(t["time"] for t in tests)
            file_time += class_time

            if class_name == "_module":
                print(f"\n  Module-level tests ({len(tests)} tests, {class_time:.1f}s total):")
            else:
                print(f"\n  Class: {class_name} ({len(tests)} tests, {class_time:.1f}s total)")
                print("  → Mark with: @pytest.mark.slow")

            for test in sorted(tests, key=lambda x: x["time"], reverse=True):
                print(f"    - {test['name']:<60} {test['time']:>6.2f}s")

        print(f"\n  File total: {len([t for tests in file_tests.values() for t in tests])} tests, {file_time:.1f}s")

    print("\n" + "=" * 100)
    print("\nSummary:")
    print(f"  Total slow tests (>20s): {len(slow_tests)}")
    print(f"  Total time in slow tests: {total_time:.1f}s")
    print(f"  Files affected: {len(by_file)}")

    # Generate actionable summary
    print("\n" + "=" * 100)
    print("\nAction Items:")
    print("-" * 100)

    # Count tests by class
    class_counts = defaultdict(int)
    for test in slow_tests:
        parts = test["classname"].split(".")
        if len(parts) > 3:
            class_key = test["classname"]
            class_counts[class_key] += 1

    if class_counts:
        print("\nClasses to mark with @pytest.mark.slow:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            parts = class_name.split(".")
            file_part = "/".join(parts[:3]) + ".py"
            class_part = "::".join(parts[3:])
            print(f"  {file_part}::{class_part} ({count} tests)")

    # Individual tests (module-level)
    individual_tests = [t for t in slow_tests if len(t["classname"].split(".")) == 3]
    if individual_tests:
        print("\nIndividual tests to mark with @pytest.mark.slow:")
        for test in individual_tests:
            print(f"  {test['file_path']}::{test['name']} ({test['time']:.1f}s)")


if __name__ == "__main__":
    main()
