/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 18, 2021.
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


#if ODBCVER >= 0x0300
void
trace_SQLGetDiagRec (int trace_leave, int retcode,
  SQLSMALLINT		  HandleType,
  SQLHANDLE		  Handle,
  SQLSMALLINT		  RecNumber,
  SQLCHAR		* SqlState,
  SQLINTEGER		* NativeErrorPtr,
  SQLCHAR		* MessageText,
  SQLSMALLINT		  BufferLength,
  SQLSMALLINT		* TextLengthPtr)
{
  /* Trace function */
  _trace_print_function (en_GetDiagRec, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handletype (HandleType);
  _trace_handle (HandleType, Handle);
  _trace_smallint (RecNumber);
  _trace_string (SqlState, SQL_NTS, NULL, TRACE_OUTPUT_SUCCESS);
  _trace_integer_p (NativeErrorPtr, TRACE_OUTPUT_SUCCESS);
  _trace_string (MessageText, BufferLength, TextLengthPtr,
      TRACE_OUTPUT_SUCCESS);
  _trace_stringlen ("SQLSMALLINT", BufferLength);
  _trace_smallint_p (TextLengthPtr, trace_leave);
}


void
trace_SQLGetDiagRecW (int trace_leave, int retcode,
  SQLSMALLINT		  HandleType,
  SQLHANDLE		  Handle,
  SQLSMALLINT		  RecNumber,
  SQLWCHAR		* SqlState,
  SQLINTEGER		* NativeErrorPtr,
  SQLWCHAR		* MessageText,
  SQLSMALLINT		  BufferLength,
  SQLSMALLINT		* TextLengthPtr)
{
  /* Trace function */
  _trace_print_function (en_GetDiagRecW, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handletype (HandleType);
  _trace_handle (HandleType, Handle);
  _trace_smallint (RecNumber);
  _trace_string_w (SqlState, SQL_NTS, NULL, TRACE_OUTPUT_SUCCESS);
  _trace_integer_p (NativeErrorPtr, TRACE_OUTPUT_SUCCESS);
  _trace_string_w (MessageText, BufferLength, TextLengthPtr,
      TRACE_OUTPUT_SUCCESS);
  _trace_stringlen ("SQLSMALLINT", BufferLength);
  _trace_smallint_p (TextLengthPtr, TRACE_OUTPUT_SUCCESS);
}
#endif
