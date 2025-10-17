/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 23, 2023.
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
trace_SQLBindParameter (int trace_leave, int retcode,
  SQLHSTMT		  StatementHandle,
  SQLUSMALLINT		  ParameterNumber,
  SQLSMALLINT		  InputOutputType,
  SQLSMALLINT		  ValueType,
  SQLSMALLINT		  ParameterType,
  SQLUINTEGER		  ColumnSize,
  SQLSMALLINT		  DecimalDigits,
  SQLPOINTER		  ParameterValuePtr,
  SQLLEN		  BufferLength,
  SQLLEN		* Strlen_or_IndPtr)
{
  /* Trace function */
  _trace_print_function (en_BindParameter, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_STMT, StatementHandle);
  _trace_smallint (ParameterNumber);
  _trace_inouttype (InputOutputType);
  _trace_c_type (ValueType);
  _trace_sql_type (ParameterType);
  _trace_uinteger (ColumnSize);
  _trace_smallint (DecimalDigits);
  _trace_pointer (ParameterValuePtr);
  _trace_len (BufferLength);
  _trace_len_p (Strlen_or_IndPtr, TRACE_OUTPUT_SUCCESS);
}
