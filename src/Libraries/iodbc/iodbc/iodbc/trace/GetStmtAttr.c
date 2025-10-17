/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 17, 2023.
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
_trace_stmtattr_type (SQLINTEGER type)
{
  char *ptr = "unknown statement attribute";

  switch (type)
    {
      _S (SQL_ATTR_APP_PARAM_DESC);
      _S (SQL_ATTR_APP_ROW_DESC);
      _S (SQL_ATTR_ASYNC_ENABLE);
      _S (SQL_ATTR_CONCURRENCY);
      _S (SQL_ATTR_CURSOR_SCROLLABLE);
      _S (SQL_ATTR_CURSOR_SENSITIVITY);
      _S (SQL_ATTR_CURSOR_TYPE);
      _S (SQL_ATTR_ENABLE_AUTO_IPD);
      _S (SQL_ATTR_FETCH_BOOKMARK_PTR);
      _S (SQL_ATTR_IMP_PARAM_DESC);
      _S (SQL_ATTR_IMP_ROW_DESC);
      _S (SQL_ATTR_KEYSET_SIZE);
      _S (SQL_ATTR_MAX_LENGTH);
      _S (SQL_ATTR_MAX_ROWS);
      _S (SQL_ATTR_NOSCAN);
      _S (SQL_ATTR_PARAMSET_SIZE);
      _S (SQL_ATTR_PARAMS_PROCESSED_PTR);
      _S (SQL_ATTR_PARAM_BIND_OFFSET_PTR);
      _S (SQL_ATTR_PARAM_BIND_TYPE);
      _S (SQL_ATTR_PARAM_OPERATION_PTR);
      _S (SQL_ATTR_PARAM_STATUS_PTR);
      _S (SQL_ATTR_QUERY_TIMEOUT);
      _S (SQL_ATTR_RETRIEVE_DATA);
      _S (SQL_ATTR_ROWS_FETCHED_PTR);
      _S (SQL_ATTR_ROW_ARRAY_SIZE);
      _S (SQL_ATTR_ROW_BIND_OFFSET_PTR);
      _S (SQL_ATTR_ROW_BIND_TYPE);
      _S (SQL_ATTR_ROW_NUMBER);
      _S (SQL_ATTR_ROW_OPERATION_PTR);
      _S (SQL_ATTR_ROW_STATUS_PTR);
      _S (SQL_ATTR_SIMULATE_CURSOR);
      _S (SQL_ATTR_USE_BOOKMARKS);
    }

  trace_emit ("\t\t%-15.15s   %d (%s)\n",
      "SQLINTEGER ", (int) type, ptr);
}


void
trace_SQLGetStmtAttr (int trace_leave, int retcode,
  SQLHSTMT		  StatementHandle,
  SQLINTEGER		  Attribute,
  SQLPOINTER		  ValuePtr,
  SQLINTEGER		  BufferLength,
  SQLINTEGER		* StringLengthPtr)
{
  /* Trace function */
  _trace_print_function (en_GetStmtAttr, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_STMT, StatementHandle);
  _trace_stmtattr_type (Attribute);
  _trace_pointer (ValuePtr);
  _trace_bufferlen (BufferLength);
  _trace_integer_p (StringLengthPtr, TRACE_OUTPUT_SUCCESS);
}


void
trace_SQLGetStmtAttrW (int trace_leave, int retcode,
  SQLHSTMT		  StatementHandle,
  SQLINTEGER		  Attribute,
  SQLPOINTER		  ValuePtr,
  SQLINTEGER		  BufferLength,
  SQLINTEGER		* StringLengthPtr)
{
  /* Trace function */
  _trace_print_function (en_GetStmtAttrW, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_STMT, StatementHandle);
  _trace_stmtattr_type (Attribute);
  _trace_pointer (ValuePtr);
  _trace_bufferlen (BufferLength);
  _trace_integer_p (StringLengthPtr, trace_leave);
}
#endif
