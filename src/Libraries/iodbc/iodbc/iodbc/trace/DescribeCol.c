/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 20, 2022.
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
_trace_desc_null (SQLSMALLINT *p, int output)
{
  char *ptr = "unknown nullable type";

  if (!p)
    {
      trace_emit ("\t\t%-15.15s * 0x0\n", "SQLSMALLINT");
      return;
    }

  if (!output)
    {
      trace_emit ("\t\t%-15.15s * %p\n", "SQLSMALLINT", p);
      return;
    }

  switch (*p)
    {
      _S (SQL_NULLABLE);
      _S (SQL_NULLABLE_UNKNOWN);
      _S (SQL_NO_NULLS);
    }

  trace_emit ("\t\t%-15.15s * %p (%s)\n", "SQLSMALLINT", p, ptr);
}


void
trace_SQLDescribeCol (int trace_leave, int retcode,
  SQLHSTMT		  StatementHandle,
  SQLSMALLINT		  ColumnNumber,
  SQLCHAR		* ColumnName,
  SQLSMALLINT		  BufferLength,
  SQLSMALLINT		* NameLengthPtr,
  SQLSMALLINT		* DataTypePtr,
  SQLULEN		* ColumnSizePtr,
  SQLSMALLINT		* DecimalDigitsPtr,
  SQLSMALLINT		* NullablePtr)
{
  /* Trace function */
  _trace_print_function (en_DescribeCol, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_STMT, StatementHandle);
  _trace_usmallint (ColumnNumber);
  _trace_string (ColumnName, BufferLength, NameLengthPtr, TRACE_OUTPUT_SUCCESS);
  _trace_stringlen ("SQLSMALLINT", BufferLength);
  _trace_smallint_p (NameLengthPtr, TRACE_OUTPUT_SUCCESS);
  _trace_sql_type_p (DataTypePtr, TRACE_OUTPUT_SUCCESS);
  _trace_ulen_p (ColumnSizePtr, TRACE_OUTPUT_SUCCESS);
  _trace_smallint_p (DecimalDigitsPtr, TRACE_OUTPUT_SUCCESS);
  _trace_desc_null (NullablePtr, TRACE_OUTPUT_SUCCESS);
}


#if ODBCVER >= 0x0300
void
trace_SQLDescribeColW (int trace_leave, int retcode,
  SQLHSTMT		  StatementHandle,
  SQLSMALLINT		  ColumnNumber,
  SQLWCHAR		* ColumnName,
  SQLSMALLINT		  BufferLength,
  SQLSMALLINT		* NameLengthPtr,
  SQLSMALLINT		* DataTypePtr,
  SQLULEN		* ColumnSizePtr,
  SQLSMALLINT		* DecimalDigitsPtr,
  SQLSMALLINT		* NullablePtr)
{
  /* Trace function */
  _trace_print_function (en_DescribeColW, trace_leave, retcode);
  
  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_STMT, StatementHandle);
  _trace_usmallint (ColumnNumber);
  _trace_string_w (ColumnName, BufferLength, NameLengthPtr, TRACE_OUTPUT_SUCCESS);
  _trace_stringlen ("SQLSMALLINT", BufferLength);
  _trace_smallint_p (NameLengthPtr, TRACE_OUTPUT_SUCCESS);
  _trace_sql_type_p (DataTypePtr, TRACE_OUTPUT_SUCCESS);
  _trace_ulen_p (ColumnSizePtr, TRACE_OUTPUT_SUCCESS);
  _trace_smallint_p (DecimalDigitsPtr, TRACE_OUTPUT_SUCCESS);
  _trace_desc_null (NullablePtr, TRACE_OUTPUT_SUCCESS);
}
#endif
