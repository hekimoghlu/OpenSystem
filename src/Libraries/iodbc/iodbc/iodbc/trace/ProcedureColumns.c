/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 19, 2024.
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
trace_SQLProcedureColumns (int trace_leave, int retcode,
  SQLHSTMT		  hstmt,
  SQLCHAR    		* szProcQualifier,
  SQLSMALLINT		  cbProcQualifier,
  SQLCHAR    		* szProcOwner,
  SQLSMALLINT		  cbProcOwner,
  SQLCHAR    		* szProcName,
  SQLSMALLINT		  cbProcName,
  SQLCHAR    		* szColumnName,
  SQLSMALLINT		  cbColumnName)
{
  /* Trace function */
  _trace_print_function (en_ProcedureColumns, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_STMT, hstmt);

  _trace_string (szProcQualifier, cbProcQualifier, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbProcQualifier);
  _trace_string (szProcOwner, cbProcOwner, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbProcOwner);
  _trace_string (szProcName, cbProcName, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbProcName);
  _trace_string (szColumnName, cbColumnName, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbColumnName);
}


#if ODBCVER >= 0x0300
void
trace_SQLProcedureColumnsW (int trace_leave, int retcode,
  SQLHSTMT		  hstmt,
  SQLWCHAR    		* szProcQualifier,
  SQLSMALLINT		  cbProcQualifier,
  SQLWCHAR    		* szProcOwner,
  SQLSMALLINT		  cbProcOwner,
  SQLWCHAR    		* szProcName,
  SQLSMALLINT		  cbProcName,
  SQLWCHAR    		* szColumnName,
  SQLSMALLINT		  cbColumnName)
{
  /* Trace function */
  _trace_print_function (en_ProcedureColumnsW, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_STMT, hstmt);

  _trace_string_w (szProcQualifier, cbProcQualifier, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbProcQualifier);
  _trace_string_w (szProcOwner, cbProcOwner, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbProcOwner);
  _trace_string_w (szProcName, cbProcName, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbProcName);
  _trace_string_w (szColumnName, cbColumnName, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbColumnName);
}
#endif
