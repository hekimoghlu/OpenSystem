/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 13, 2023.
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
_trace_setpos_irow (SQLSETPOSIROW i)
{
#ifdef _WIN64
  trace_emit ("\t\t%-15.15s   %I64d\n", "SQLSETPOSIROW", (INT64) i);
#else
  trace_emit ("\t\t%-15.15s   %ld\n", "SQLSETPOSIROW", (long) i);
#endif
}


void
_trace_setpos_oper (SQLUSMALLINT type)
{
  char *ptr = "unknown operation";

  switch (type)
    {
      _S (SQL_POSITION);
      _S (SQL_REFRESH);
      _S (SQL_UPDATE);
      _S (SQL_DELETE);
      _S (SQL_ADD);
#if ODBCVER >= 0x0300
      _S (SQL_UPDATE_BY_BOOKMARK);
      _S (SQL_DELETE_BY_BOOKMARK);
      _S (SQL_FETCH_BY_BOOKMARK);
#endif
    }

  trace_emit ("\t\t%-15.15s   %d (%s)\n", "SQLUSMALLINT", (int) type, ptr);
}


void
_trace_setpos_lock (SQLUSMALLINT type)
{
  char *ptr = "unknown lock type";

  switch (type)
    {
      _S (SQL_LOCK_NO_CHANGE);
      _S (SQL_LOCK_EXCLUSIVE);
      _S (SQL_LOCK_UNLOCK);
    }

  trace_emit ("\t\t%-15.15s   %d (%s)\n", "SQLUSMALLINT", (int) type, ptr);
}


void
trace_SQLSetPos (int trace_leave, int retcode,
  SQLHSTMT		  hstmt,
  SQLSETPOSIROW		  irow,
  SQLUSMALLINT		  fOption,
  SQLUSMALLINT		  fLock)
{
  /* Trace function */
  _trace_print_function (en_SetPos, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_STMT, hstmt);
  _trace_setpos_irow (irow);
  _trace_setpos_oper (fOption);
  _trace_setpos_lock (fLock);
}
