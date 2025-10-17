/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 29, 2021.
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
trace_SQLSetCursorName (int trace_leave, int retcode,
  SQLHSTMT		  hstmt,
  SQLCHAR		* szCursor,
  SQLSMALLINT		  cbCursor)
{
  /* Trace function */
  _trace_print_function (en_SetCursorName, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_STMT, hstmt);
  _trace_string (szCursor, cbCursor, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbCursor);
}


#if ODBCVER >= 0x0300
void
trace_SQLSetCursorNameW (int trace_leave, int retcode,
  SQLHSTMT		  hstmt,
  SQLWCHAR 		* szCursor,
  SQLSMALLINT		  cbCursor)
{
  /* Trace function */
  _trace_print_function (en_SetCursorNameW, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_STMT, hstmt);
  _trace_string_w (szCursor, cbCursor, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbCursor);
}
#endif
