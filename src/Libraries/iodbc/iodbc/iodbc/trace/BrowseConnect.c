/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 1, 2024.
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
trace_SQLBrowseConnect (int trace_leave, int retcode,
  SQLHDBC		  ConnectionHandle,
  SQLCHAR 		* InConnectionString,
  SQLSMALLINT		  StringLength1,
  SQLCHAR		* OutConnectionString,
  SQLSMALLINT		  BufferLength,
  SQLSMALLINT		* StringLength2Ptr)
{
  /* Trace function */
  _trace_print_function (en_BrowseConnect, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_DBC, ConnectionHandle);
  _trace_string (InConnectionString, StringLength1, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", StringLength1);
  _trace_string (OutConnectionString, BufferLength, StringLength2Ptr,
      TRACE_OUTPUT_SUCCESS);
  _trace_smallint (BufferLength);
  _trace_smallint_p (StringLength2Ptr, TRACE_OUTPUT_SUCCESS);
}


#if ODBCVER >= 0x0300
void
trace_SQLBrowseConnectW (int trace_leave, int retcode,
  SQLHDBC		  ConnectionHandle,
  SQLWCHAR 		* InConnectionString,
  SQLSMALLINT		  StringLength1,
  SQLWCHAR		* OutConnectionString,
  SQLSMALLINT		  BufferLength,
  SQLSMALLINT		* StringLength2Ptr)
{
  /* Trace function */
  _trace_print_function (en_BrowseConnectW, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_DBC, ConnectionHandle);
  _trace_string_w (InConnectionString, StringLength1, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", StringLength1);
  _trace_string_w (OutConnectionString, BufferLength, StringLength2Ptr,
      TRACE_OUTPUT_SUCCESS);
  _trace_smallint (BufferLength);
  _trace_smallint_p (StringLength2Ptr, TRACE_OUTPUT_SUCCESS);
}
#endif
