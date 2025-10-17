/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 22, 2023.
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
trace_SQLError (int trace_leave, int retcode,
  SQLHENV		  henv,
  SQLHDBC		  hdbc,
  SQLHSTMT		  hstmt,
  SQLCHAR  		* szSqlstate,
  SQLINTEGER  		* pfNativeError,
  SQLCHAR  		* szErrorMsg,
  SQLSMALLINT		  cbErrorMsgMax,
  SQLSMALLINT  		* pcbErrorMsg)
{
  /* Trace function */
  _trace_print_function (en_Error, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_ENV, henv);
  _trace_handle (SQL_HANDLE_DBC, hdbc);
  _trace_handle (SQL_HANDLE_STMT, hstmt);
  _trace_string (szSqlstate, SQL_NTS, NULL, TRACE_OUTPUT_SUCCESS);
  _trace_integer_p (pfNativeError, TRACE_OUTPUT_SUCCESS);
  _trace_string (szErrorMsg, cbErrorMsgMax, pcbErrorMsg, TRACE_OUTPUT_SUCCESS);
  _trace_stringlen ("SQLINTEGER", cbErrorMsgMax);
  _trace_smallint_p (pcbErrorMsg, TRACE_OUTPUT_SUCCESS);
}


#if ODBCVER >= 0x0300
void
trace_SQLErrorW (int trace_leave, int retcode,
  SQLHENV		  henv,
  SQLHDBC		  hdbc,
  SQLHSTMT		  hstmt,
  SQLWCHAR  		* szSqlstate,
  SQLINTEGER  		* pfNativeError,
  SQLWCHAR  		* szErrorMsg,
  SQLSMALLINT		  cbErrorMsgMax,
  SQLSMALLINT  		* pcbErrorMsg)
{
  /* Trace function */
  _trace_print_function (en_Error, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_ENV, henv);
  _trace_handle (SQL_HANDLE_DBC, hdbc);
  _trace_handle (SQL_HANDLE_STMT, hstmt);
  _trace_string_w (szSqlstate, SQL_NTS, NULL, TRACE_OUTPUT_SUCCESS);
  _trace_integer_p (pfNativeError, trace_leave);
  _trace_string_w (szErrorMsg, cbErrorMsgMax, pcbErrorMsg, TRACE_OUTPUT_SUCCESS);
  _trace_stringlen ("SQLINTEGER", cbErrorMsgMax);
  _trace_smallint_p (pcbErrorMsg, TRACE_OUTPUT_SUCCESS);
}
#endif
