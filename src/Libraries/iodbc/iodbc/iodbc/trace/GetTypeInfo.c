/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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
_trace_typeinfo (SQLSMALLINT type)
{
  char *ptr = "unknown type";

  switch (type)
    {
      _S (SQL_ALL_TYPES);
      _S (SQL_BIGINT);
      _S (SQL_BINARY);
      _S (SQL_BIT);
      _S (SQL_CHAR);
#if (ODBCVER < 0x0300)
      _S (SQL_DATE);
#else
      _S (SQL_DATETIME);
#endif
      _S (SQL_DECIMAL);
      _S (SQL_DOUBLE);
      _S (SQL_FLOAT);
#if (ODBCVER >= 0x0350)
      _S (SQL_GUID);
#endif
      _S (SQL_INTEGER);
      _S (SQL_LONGVARBINARY);
      _S (SQL_LONGVARCHAR);
      _S (SQL_NUMERIC);
      _S (SQL_REAL);
      _S (SQL_SMALLINT);
#if (ODBCVER < 0x0300)
      _S (SQL_TIME);
#else
      _S (SQL_INTERVAL);
#endif
      _S (SQL_TIMESTAMP);
      _S (SQL_TINYINT);
#if ODBCVER >= 0x0300
      _S (SQL_TYPE_DATE);
      _S (SQL_TYPE_TIME);
      _S (SQL_TYPE_TIMESTAMP);
#endif
      _S (SQL_VARBINARY);
      _S (SQL_VARCHAR);
      _S (SQL_WCHAR);
      _S (SQL_WLONGVARCHAR);
      _S (SQL_WVARCHAR);
    }

  trace_emit ("\t\t%-15.15s   %d (%s)\n", "SQLSMALLINT ", (int) type, ptr);
}


void 
trace_SQLGetTypeInfo (int trace_leave, int retcode,
  SQLHSTMT		  hstmt,
  SQLSMALLINT		  fSqlType)
{
  /* Trace function */
  _trace_print_function (en_GetTypeInfo, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_STMT, hstmt);
  _trace_typeinfo (fSqlType);
}


#if ODBCVER >= 0x0300
void 
trace_SQLGetTypeInfoW (int trace_leave, int retcode,
  SQLHSTMT		  hstmt,
  SQLSMALLINT		  fSqlType)
{
  /* Trace function */
  _trace_print_function (en_GetTypeInfoW, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_STMT, hstmt);
  _trace_typeinfo (fSqlType);
}
#endif
