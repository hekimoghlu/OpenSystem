/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 15, 2025.
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
_trace_descfield_type (SQLSMALLINT type)
{
  char *ptr = "unknown field identifier";

  switch (type)
    {
      _S (SQL_DESC_ALLOC_TYPE);
      _S (SQL_DESC_ARRAY_SIZE);
      _S (SQL_DESC_ARRAY_STATUS_PTR);
      _S (SQL_DESC_AUTO_UNIQUE_VALUE);
      _S (SQL_DESC_BASE_COLUMN_NAME);
      _S (SQL_DESC_BASE_TABLE_NAME);
      _S (SQL_DESC_BIND_OFFSET_PTR);
      _S (SQL_DESC_BIND_TYPE);
      _S (SQL_DESC_CASE_SENSITIVE);
      _S (SQL_DESC_CATALOG_NAME);
      _S (SQL_DESC_CONCISE_TYPE);
      _S (SQL_DESC_COUNT);
      _S (SQL_DESC_DATA_PTR);
      _S (SQL_DESC_DATETIME_INTERVAL_CODE);
      _S (SQL_DESC_DATETIME_INTERVAL_PRECISION);
      _S (SQL_DESC_DISPLAY_SIZE);
      _S (SQL_DESC_FIXED_PREC_SCALE);
      _S (SQL_DESC_INDICATOR_PTR);
      _S (SQL_DESC_LABEL);
      _S (SQL_DESC_LENGTH);
      _S (SQL_DESC_LITERAL_PREFIX);
      _S (SQL_DESC_LITERAL_SUFFIX);
      _S (SQL_DESC_LOCAL_TYPE_NAME);
      _S (SQL_DESC_MAXIMUM_SCALE);
      _S (SQL_DESC_MINIMUM_SCALE);
      _S (SQL_DESC_NAME);
      _S (SQL_DESC_NULLABLE);
      _S (SQL_DESC_NUM_PREC_RADIX);
      _S (SQL_DESC_OCTET_LENGTH);
      _S (SQL_DESC_OCTET_LENGTH_PTR);
      _S (SQL_DESC_PARAMETER_TYPE);
      _S (SQL_DESC_PRECISION);
      _S (SQL_DESC_ROWS_PROCESSED_PTR);
      _S (SQL_DESC_SCALE);
      _S (SQL_DESC_SCHEMA_NAME);
      _S (SQL_DESC_SEARCHABLE);
      _S (SQL_DESC_TABLE_NAME);
      _S (SQL_DESC_TYPE);
      _S (SQL_DESC_TYPE_NAME);
      _S (SQL_DESC_UNNAMED);
      _S (SQL_DESC_UNSIGNED);
      _S (SQL_DESC_UPDATABLE);

#if (ODBCVER >= 0x0350)
      _S (SQL_DESC_ROWVER);
#endif
    }

  trace_emit ("\t\t%-15.15s   %d (%s)\n", "SQLUSMALLINT ", (int) type, ptr);
}


void
trace_SQLGetDescField (int trace_leave, int retcode,
  SQLHDESC		  DescriptorHandle,
  SQLSMALLINT		  RecNumber,
  SQLSMALLINT		  FieldIdentifier,
  SQLPOINTER		  ValuePtr,
  SQLINTEGER		  BufferLength,
  SQLINTEGER		* StringLengthPtr)
{
  /* Trace function */
  _trace_print_function (en_GetDescField, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_DESC, DescriptorHandle);
  _trace_smallint (RecNumber);
  _trace_descfield_type (FieldIdentifier);
  _trace_pointer (ValuePtr);
  _trace_bufferlen (BufferLength);
  _trace_integer_p (StringLengthPtr, TRACE_OUTPUT_SUCCESS);
}


void
trace_SQLGetDescFieldW (int trace_leave, int retcode,
  SQLHDESC		  DescriptorHandle,
  SQLSMALLINT		  RecNumber,
  SQLSMALLINT		  FieldIdentifier,
  SQLPOINTER		  ValuePtr,
  SQLINTEGER		  BufferLength,
  SQLINTEGER		* StringLengthPtr)
{
  /* Trace function */
  _trace_print_function (en_GetDescFieldW, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_DESC, DescriptorHandle);
  _trace_smallint (RecNumber);
  _trace_descfield_type (FieldIdentifier);
  _trace_pointer (ValuePtr);
  _trace_bufferlen (BufferLength);
  _trace_integer_p (StringLengthPtr, TRACE_OUTPUT_SUCCESS);
}
#endif
