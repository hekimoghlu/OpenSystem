/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 1, 2023.
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
_trace_connopt_type (SQLUSMALLINT type)
{
  char *ptr = "unknown connection attribute";

  switch (type)
    {
      _S (SQL_ACCESS_MODE);
      _S (SQL_AUTOCOMMIT);
      _S (SQL_CURRENT_QUALIFIER);
      _S (SQL_LOGIN_TIMEOUT);
      _S (SQL_ODBC_CURSORS);
      _S (SQL_OPT_TRACE);
      _S (SQL_OPT_TRACEFILE);
      _S (SQL_PACKET_SIZE);
      _S (SQL_QUIET_MODE);
      _S (SQL_TRANSLATE_DLL);
      _S (SQL_TRANSLATE_OPTION);
      _S (SQL_TXN_ISOLATION);

      /* 2.0 Driver Manager also allows statement options at this time */
      _S (SQL_ASYNC_ENABLE);
      _S (SQL_BIND_TYPE);
      _S (SQL_CONCURRENCY);
      _S (SQL_CURSOR_TYPE);
      _S (SQL_KEYSET_SIZE);
      _S (SQL_MAX_LENGTH);
      _S (SQL_MAX_ROWS);
      _S (SQL_NOSCAN);
      _S (SQL_QUERY_TIMEOUT);
      _S (SQL_RETRIEVE_DATA);
      _S (SQL_ROWSET_SIZE);
      _S (SQL_SIMULATE_CURSOR);
      _S (SQL_USE_BOOKMARKS);
    }

  trace_emit ("\t\t%-15.15s   %ld (%s)\n", "SQLUSMALLINT ", (int) type, ptr);
}


void
trace_SQLGetConnectOption (int trace_leave, int retcode,
  SQLHDBC		  hdbc,
  SQLUSMALLINT		  fOption,
  SQLPOINTER		  pvParam)
{
  /* Trace function */
  _trace_print_function (en_GetConnectOption, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_DBC, hdbc);
  _trace_connopt_type (fOption);
  _trace_pointer (pvParam);
}


#if ODBCVER >= 0x0300
void
trace_SQLGetConnectOptionW (int trace_leave, int retcode,
  SQLHDBC		  hdbc,
  SQLUSMALLINT		  fOption,
  SQLPOINTER		  pvParam)
{
  /* Trace function */
  _trace_print_function (en_GetConnectOptionW, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_DBC, hdbc);
  _trace_connopt_type (fOption);
  _trace_pointer (pvParam);
}
#endif
