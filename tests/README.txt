For simplicity. tests generally are run from the parent directory. This ensures
the runtime environment matches the script being tested. If local files are
needed, the path should determined via invocation path name (e.g., see the
resolve_path function in ../glue_helpers.py).

For example,

    $ PYTHONPATH=/tmp python test_main.py
    Traceback (most recent call last):
      File "test_main.py", line 14, in <module>
        from unittest_wrapper import TestWrapper
    ModuleNotFoundError: No module named 'unittest_wrapper'

versus

    $ python tests/test_main.py
    Ran 2 tests in 0.402s
    
    OK

