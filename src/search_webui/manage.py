#!/usr/bin/env python
import os
import sys
lib_path = os.path.abspath('../..')
sys.path.insert(0, lib_path)
prj_name  = sys.argv[-1]
assert prj_name == 'apparel' or prj_name == 'footwear'
if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "search_webui.settings")

    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv[:-1])
