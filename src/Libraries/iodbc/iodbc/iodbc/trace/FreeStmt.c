/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 21, 2025.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include "trace.h"


void
_trace_freestmt_option (int option)
{
  char *ptr = "invalid option";

  switch (option)
    {
      _S (SQL_CLOSE);
      _S (SQL_DROP);
      _S (SQL_UNBIND);
      _S (SQL_RESET_PARAMS);
    }

  trace_emit ("\t\t%-15.15s   %d (%s)\n", "SQLUSMALLINT", (int) option, ptr);
}


void
trace_SQLFreeStmt (int trace_leave, int retcode, 
  SQLHSTMT		   hstmt, 
  SQLUSMALLINT		   Option)
{
  /* Trace function */
  _trace_print_function (en_FreeStmt, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_STMT, hstmt);
  _trace_freestmt_option (Option);
}
