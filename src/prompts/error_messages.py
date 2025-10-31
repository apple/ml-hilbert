#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
################################################################
#               Lean Verifier Error Messages
################################################################

FULL_ERROR_MESSAGE = """

The provided proof:
```lean4 
{proof}
```

{error_lines_message}
"""

ERROR_LINE_MESSAGE = """
Error message from Lean: {error_message}
The error was encountered while trying to process the following lines:
{error_lines}
{current_state}
"""

CURRENT_STATE_MESSAGE = """
Current proof state when error occurred:
{goal_state}
"""
