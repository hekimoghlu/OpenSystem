/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 14, 2024.
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
trace_SQLConnect (int trace_leave, int retcode,
  SQLHDBC		  hdbc,
  SQLCHAR  		* szDSN,
  SQLSMALLINT		  cbDSN,
  SQLCHAR   		* szUID,
  SQLSMALLINT		  cbUID,
  SQLCHAR   		* szAuthStr,
  SQLSMALLINT		  cbAuthStr)
{
  /* Trace function */
  _trace_print_function (en_Connect, trace_leave, retcode);

  /* Hide plaintext passwords */
  szAuthStr = (SQLCHAR *) "****";

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_DBC, hdbc);
  _trace_string (szDSN, cbDSN, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbDSN);
  _trace_string (szUID, cbUID, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbDSN);
  _trace_string (szAuthStr, SQL_NTS, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbAuthStr);
}


#if ODBCVER >= 0x0300
void
trace_SQLConnectW (int trace_leave, int retcode,
  SQLHDBC		  hdbc,
  SQLWCHAR  		* szDSN,
  SQLSMALLINT		  cbDSN,
  SQLWCHAR  		* szUID,
  SQLSMALLINT		  cbUID,
  SQLWCHAR  		* szAuthStr,
  SQLSMALLINT		  cbAuthStr)
{
  /* Trace function */
  _trace_print_function (en_ConnectW, trace_leave, retcode);

  /* Hide plaintext passwords */
  szAuthStr = (SQLWCHAR *) L"****";

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_DBC, hdbc);
  _trace_string_w (szDSN, cbDSN, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbDSN);
  _trace_string_w (szUID, cbUID, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbDSN);
  _trace_string_w (szAuthStr, SQL_NTS, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbAuthStr);
}
#endif
