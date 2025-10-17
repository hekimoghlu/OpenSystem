/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 11, 2024.
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
trace_SQLPrepare (int trace_leave, int retcode,
  SQLHSTMT		  StatementHandle,
  SQLCHAR 		* StatementText,
  SQLINTEGER		  TextLength)
{
  /* Trace function */
  _trace_print_function (en_Prepare, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_STMT, StatementHandle);
  _trace_string (StatementText, TextLength, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLINTEGER", TextLength);
}


#if ODBCVER >= 0x0300
void
trace_SQLPrepareW (int trace_leave, int retcode,
  SQLHSTMT		  StatementHandle,
  SQLWCHAR 		* StatementText,
  SQLINTEGER		  TextLength)
{
  /* Trace function */
  _trace_print_function (en_PrepareW, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_STMT, StatementHandle);
  _trace_string_w (StatementText, TextLength, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLINTEGER", TextLength);
}
#endif
