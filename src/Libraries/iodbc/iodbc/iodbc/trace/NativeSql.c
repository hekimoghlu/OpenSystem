/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 14, 2024.
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
trace_SQLNativeSql (int trace_leave, int retcode,
  SQLHDBC		  hdbc,
  SQLCHAR		* InStatementText,
  SQLINTEGER		  TextLength1,
  SQLCHAR		* OutStatementText,
  SQLINTEGER		  BufferLength,
  SQLINTEGER		* TextLength2Ptr)
{
  SQLSMALLINT len = 0;

  if (TextLength2Ptr)
    len = *TextLength2Ptr;

  /* Trace function */
  _trace_print_function (en_NativeSql, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_DBC, hdbc);
  _trace_string (InStatementText, TextLength1, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLINTEGER", TextLength1);
  _trace_string (OutStatementText, BufferLength, &len, TRACE_OUTPUT_SUCCESS);
  _trace_stringlen ("SQLINTEGER", BufferLength);
  _trace_integer_p (TextLength2Ptr, TRACE_OUTPUT_SUCCESS);
}


#if ODBCVER >= 0x0300
void 
trace_SQLNativeSqlW (int trace_leave, int retcode,
  SQLHDBC		  hdbc,
  SQLWCHAR		* InStatementText,
  SQLINTEGER		  TextLength1,
  SQLWCHAR		* OutStatementText,
  SQLINTEGER		  BufferLength,
  SQLINTEGER		* TextLength2Ptr)
{
  SQLSMALLINT len = 0;

  if (TextLength2Ptr)
    len = *TextLength2Ptr;

  /* Trace function */
  _trace_print_function (en_NativeSqlW, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_DBC, hdbc);
  _trace_string_w (InStatementText, TextLength1, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLINTEGER", TextLength1);
  _trace_string_w (OutStatementText, BufferLength, &len, TRACE_OUTPUT_SUCCESS);
  _trace_stringlen ("SQLINTEGER", BufferLength);
  _trace_integer_p (TextLength2Ptr, TRACE_OUTPUT_SUCCESS);
}
#endif
