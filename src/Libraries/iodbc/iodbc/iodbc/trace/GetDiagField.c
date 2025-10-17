/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 20, 2023.
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
_trace_diag_type (SQLSMALLINT type)
{
  char *ptr = "unknown diag identifier";

  switch (type)
    {
      _S (SQL_DIAG_CLASS_ORIGIN);
      _S (SQL_DIAG_COLUMN_NUMBER);
      _S (SQL_DIAG_CONNECTION_NAME);
      _S (SQL_DIAG_CURSOR_ROW_COUNT);
      _S (SQL_DIAG_DYNAMIC_FUNCTION);
      _S (SQL_DIAG_DYNAMIC_FUNCTION_CODE);
      _S (SQL_DIAG_MESSAGE_TEXT);
      _S (SQL_DIAG_NATIVE);
      _S (SQL_DIAG_NUMBER);
      _S (SQL_DIAG_RETURNCODE);
      _S (SQL_DIAG_ROW_COUNT);
      _S (SQL_DIAG_ROW_NUMBER);
      _S (SQL_DIAG_SERVER_NAME);
      _S (SQL_DIAG_SQLSTATE);
      _S (SQL_DIAG_SUBCLASS_ORIGIN);
    }

  trace_emit ("\t\t%-15.15s   %d (%s)\n", "SQLUSMALLINT ", (int) type, ptr);
}

void
trace_SQLGetDiagField (int trace_leave, int retcode,
  SQLSMALLINT		  HandleType,
  SQLHANDLE		  Handle,
  SQLSMALLINT		  RecNumber,
  SQLSMALLINT		  DiagIdentifier,
  SQLPOINTER		  DiagInfoPtr,
  SQLSMALLINT		  BufferLength,
  SQLSMALLINT		* StringLengthPtr)
{
  /* Trace function */
  _trace_print_function (en_GetDiagField, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handletype (HandleType);
  _trace_handle (HandleType, Handle);
  _trace_smallint (RecNumber);
  _trace_diag_type (DiagIdentifier);
  _trace_pointer (DiagInfoPtr);
  _trace_bufferlen ((SQLINTEGER) BufferLength);
  _trace_smallint_p (StringLengthPtr, TRACE_OUTPUT_SUCCESS);
}


void
trace_SQLGetDiagFieldW (int trace_leave, int retcode,
  SQLSMALLINT		  HandleType,
  SQLHANDLE		  Handle,
  SQLSMALLINT		  RecNumber,
  SQLSMALLINT		  DiagIdentifier,
  SQLPOINTER		  DiagInfoPtr,
  SQLSMALLINT		  BufferLength,
  SQLSMALLINT		* StringLengthPtr)
{
  /* Trace function */
  _trace_print_function (en_GetDiagFieldW, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handletype (HandleType);
  _trace_handle (HandleType, Handle);
  _trace_smallint (RecNumber);
  _trace_diag_type (DiagIdentifier);
  _trace_pointer (DiagInfoPtr);
  _trace_bufferlen ((SQLINTEGER) BufferLength);
  _trace_smallint_p (StringLengthPtr, trace_leave);
}
#endif
