/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 28, 2023.
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
trace_SQLForeignKeys (int trace_leave, int retcode,
  SQLHSTMT		  hstmt,
  SQLCHAR    		* szPkTableQualifier,
  SQLSMALLINT		  cbPkTableQualifier,
  SQLCHAR    		* szPkTableOwner,
  SQLSMALLINT		  cbPkTableOwner,
  SQLCHAR    		* szPkTableName,
  SQLSMALLINT		  cbPkTableName,
  SQLCHAR    		* szFkTableQualifier,
  SQLSMALLINT		  cbFkTableQualifier,
  SQLCHAR    		* szFkTableOwner,
  SQLSMALLINT		  cbFkTableOwner,
  SQLCHAR    		* szFkTableName,
  SQLSMALLINT		  cbFkTableName)
{
  /* Trace function */
  _trace_print_function (en_ForeignKeys, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_STMT, hstmt);

  _trace_string (szPkTableQualifier, cbPkTableQualifier, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbPkTableQualifier);
  _trace_string (szPkTableOwner, cbPkTableOwner, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbPkTableOwner);
  _trace_string (szPkTableName, cbPkTableName, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbPkTableName);

  _trace_string (szFkTableQualifier, cbFkTableQualifier, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbFkTableQualifier);
  _trace_string (szFkTableOwner, cbFkTableOwner, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbFkTableOwner);
  _trace_string (szFkTableName, cbFkTableName, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbFkTableName);
}


#if ODBCVER >= 0x0300
void
trace_SQLForeignKeysW (int trace_leave, int retcode,
  SQLHSTMT		  hstmt,
  SQLWCHAR    		* szPkTableQualifier,
  SQLSMALLINT		  cbPkTableQualifier,
  SQLWCHAR    		* szPkTableOwner,
  SQLSMALLINT		  cbPkTableOwner,
  SQLWCHAR    		* szPkTableName,
  SQLSMALLINT		  cbPkTableName,
  SQLWCHAR    		* szFkTableQualifier,
  SQLSMALLINT		  cbFkTableQualifier,
  SQLWCHAR    		* szFkTableOwner,
  SQLSMALLINT		  cbFkTableOwner,
  SQLWCHAR    		* szFkTableName,
  SQLSMALLINT		  cbFkTableName)
{
  /* Trace function */
  _trace_print_function (en_ForeignKeysW, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_STMT, hstmt);

  _trace_string_w (szPkTableQualifier, cbPkTableQualifier, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbPkTableQualifier);
  _trace_string_w (szPkTableOwner, cbPkTableOwner, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbPkTableOwner);
  _trace_string_w (szPkTableName, cbPkTableName, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbPkTableName);

  _trace_string_w (szFkTableQualifier, cbFkTableQualifier, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbFkTableQualifier);
  _trace_string_w (szFkTableOwner, cbFkTableOwner, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbFkTableOwner);
  _trace_string_w (szFkTableName, cbFkTableName, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbFkTableName);
}
#endif
