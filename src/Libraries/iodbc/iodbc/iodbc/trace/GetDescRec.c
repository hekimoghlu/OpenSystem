/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 18, 2024.
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
trace_SQLGetDescRec (int trace_leave, int retcode,
  SQLHDESC		  DescriptorHandle,
  SQLSMALLINT		  RecNumber,
  SQLCHAR		* Name,
  SQLSMALLINT		  BufferLength,
  SQLSMALLINT		* StringLengthPtr,
  SQLSMALLINT		* TypePtr,
  SQLSMALLINT		* SubTypePtr,
  SQLLEN		* LengthPtr,
  SQLSMALLINT		* PrecisionPtr,
  SQLSMALLINT		* ScalePtr,
  SQLSMALLINT		* NullablePtr)
{
  /* Trace function */
  _trace_print_function (en_GetDescRec, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_DESC, DescriptorHandle);
  _trace_smallint (RecNumber);
  _trace_string (Name, BufferLength, StringLengthPtr, TRACE_OUTPUT_SUCCESS);
  _trace_stringlen ("SQLSMALLINT", BufferLength);
  _trace_smallint_p (StringLengthPtr, TRACE_OUTPUT_SUCCESS);
  _trace_sql_type_p (TypePtr, TRACE_OUTPUT_SUCCESS);
  _trace_sql_subtype (TypePtr, SubTypePtr, TRACE_OUTPUT_SUCCESS);
  _trace_len_p (LengthPtr, TRACE_OUTPUT_SUCCESS);
  _trace_smallint_p (PrecisionPtr, TRACE_OUTPUT_SUCCESS);
  _trace_smallint_p (ScalePtr, TRACE_OUTPUT_SUCCESS);
  _trace_desc_null (NullablePtr, TRACE_OUTPUT_SUCCESS);
}


void
trace_SQLGetDescRecW (int trace_leave, int retcode,
  SQLHDESC		  DescriptorHandle,
  SQLSMALLINT		  RecNumber,
  SQLWCHAR		* Name,
  SQLSMALLINT		  BufferLength,
  SQLSMALLINT		* StringLengthPtr,
  SQLSMALLINT		* TypePtr,
  SQLSMALLINT		* SubTypePtr,
  SQLLEN		* LengthPtr,
  SQLSMALLINT		* PrecisionPtr,
  SQLSMALLINT		* ScalePtr,
  SQLSMALLINT		* NullablePtr)
{
  /* Trace function */
  _trace_print_function (en_GetDescRecW, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_DESC, DescriptorHandle);
  _trace_smallint (RecNumber);
  _trace_string_w (Name, BufferLength, StringLengthPtr, TRACE_OUTPUT_SUCCESS);
  _trace_stringlen ("SQLSMALLINT", BufferLength);
  _trace_smallint_p (StringLengthPtr, TRACE_OUTPUT_SUCCESS);
  _trace_sql_type_p (TypePtr, TRACE_OUTPUT_SUCCESS);
  _trace_sql_subtype (TypePtr, SubTypePtr, TRACE_OUTPUT_SUCCESS);
  _trace_len_p (LengthPtr, TRACE_OUTPUT_SUCCESS);
  _trace_smallint_p (PrecisionPtr, TRACE_OUTPUT_SUCCESS);
  _trace_smallint_p (ScalePtr, TRACE_OUTPUT_SUCCESS);
  _trace_desc_null (NullablePtr, TRACE_OUTPUT_SUCCESS);
}
#endif
