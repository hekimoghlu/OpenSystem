/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 4, 2025.
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
_trace_spcols_type (SQLUSMALLINT type)
{
  char *ptr = "unknown column type";

  switch (type)
    {
      _S (SQL_BEST_ROWID);
      _S (SQL_ROWVER);
    }

  trace_emit ("\t\t%-15.15s   %d (%s)\n", "SQLUSMALLINT ", (int) type, ptr);
}


void
_trace_spcols_scope (SQLUSMALLINT type)
{
  char *ptr = "unknown scope";

  switch (type)
    {
      _S (SQL_SCOPE_CURROW);
      _S (SQL_SCOPE_TRANSACTION);
      _S (SQL_SCOPE_SESSION);
    }

  trace_emit ("\t\t%-15.15s   %d (%s)\n", "SQLUSMALLINT ", (int) type, ptr);
}


void
_trace_spcols_null (SQLUSMALLINT type)
{
  char *ptr = "unknown option";

  switch (type)
    {
      _S (SQL_NO_NULLS);
      _S (SQL_NULLABLE);
    }

  trace_emit ("\t\t%-15.15s   %d (%s)\n", "SQLUSMALLINT ", (int) type, ptr);
}


void
trace_SQLSpecialColumns (int trace_leave, int retcode,
  SQLHSTMT		  hstmt,
  SQLUSMALLINT		  fColType,
  SQLCHAR    		* szTableQualifier,
  SQLSMALLINT		  cbTableQualifier,
  SQLCHAR    		* szTableOwner,
  SQLSMALLINT		  cbTableOwner,
  SQLCHAR    		* szTableName,
  SQLSMALLINT		  cbTableName,
  SQLUSMALLINT		  fScope,
  SQLUSMALLINT		  fNullable)
{
  /* Trace function */
  _trace_print_function (en_SpecialColumns, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_STMT, hstmt);
  _trace_spcols_type (fColType);
  _trace_string (szTableQualifier, cbTableQualifier, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbTableQualifier);
  _trace_string (szTableOwner, cbTableOwner, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbTableOwner);
  _trace_string (szTableName, cbTableName, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbTableName);
  _trace_spcols_scope (fScope);
  _trace_spcols_null (fNullable);
}


#if ODBCVER >= 0x0300
void
trace_SQLSpecialColumnsW (int trace_leave, int retcode,
  SQLHSTMT		  hstmt,
  SQLUSMALLINT		  fColType,
  SQLWCHAR    		* szTableQualifier,
  SQLSMALLINT		  cbTableQualifier,
  SQLWCHAR    		* szTableOwner,
  SQLSMALLINT		  cbTableOwner,
  SQLWCHAR    		* szTableName,
  SQLSMALLINT		  cbTableName,
  SQLUSMALLINT		  fScope,
  SQLUSMALLINT		  fNullable)
{
  /* Trace function */
  _trace_print_function (en_SpecialColumnsW, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_STMT, hstmt);
  _trace_spcols_type (fColType);
  _trace_string_w (szTableQualifier, cbTableQualifier, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbTableQualifier);
  _trace_string_w (szTableOwner, cbTableOwner, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbTableOwner);
  _trace_string_w (szTableName, cbTableName, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbTableName);
  _trace_spcols_scope (fScope);
  _trace_spcols_null (fNullable);
}
#endif
