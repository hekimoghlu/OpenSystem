/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 2, 2024.
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
_trace_stats_unique (SQLUSMALLINT type)
{
  char *ptr = "unknown option";

  switch (type)
    {
      _S (SQL_INDEX_ALL);
      _S (SQL_INDEX_UNIQUE);
    }

  trace_emit ("\t\t%-15.15s   %d (%s)\n", "SQLUSMALLINT ", (int) type, ptr);
}


void
_trace_stats_accuracy (SQLUSMALLINT type)
{
  char *ptr = "unknown option";

  switch (type)
    {
      _S (SQL_ENSURE);
      _S (SQL_QUICK);
    }

  trace_emit ("\t\t%-15.15s   %d (%s)\n", "SQLUSMALLINT ", (int) type, ptr);
}


void
trace_SQLStatistics (int trace_leave, int retcode,
  SQLHSTMT		  hstmt,
  SQLCHAR    		* szTableQualifier,
  SQLSMALLINT		  cbTableQualifier,
  SQLCHAR    		* szTableOwner,
  SQLSMALLINT		  cbTableOwner,
  SQLCHAR    		* szTableName,
  SQLSMALLINT		  cbTableName,
  SQLUSMALLINT		  fUnique,
  SQLUSMALLINT		  fAccuracy)
{
  /* Trace function */
  _trace_print_function (en_Statistics, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_STMT, hstmt);
  _trace_string (szTableQualifier, cbTableQualifier, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbTableQualifier);
  _trace_string (szTableOwner, cbTableOwner, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbTableOwner);
  _trace_string (szTableName, cbTableName, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbTableName);
  _trace_stats_unique (fUnique);
  _trace_stats_accuracy (fAccuracy);
}


#if ODBCVER >= 0x0300
void
trace_SQLStatisticsW (int trace_leave, int retcode,
  SQLHSTMT		  hstmt,
  SQLWCHAR    		* szTableQualifier,
  SQLSMALLINT		  cbTableQualifier,
  SQLWCHAR    		* szTableOwner,
  SQLSMALLINT		  cbTableOwner,
  SQLWCHAR    		* szTableName,
  SQLSMALLINT		  cbTableName,
  SQLUSMALLINT		  fUnique,
  SQLUSMALLINT		  fAccuracy)
{
  /* Trace function */
  _trace_print_function (en_Statistics, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_STMT, hstmt);
  _trace_string_w (szTableQualifier, cbTableQualifier, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbTableQualifier);
  _trace_string_w (szTableOwner, cbTableOwner, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbTableOwner);
  _trace_string_w (szTableName, cbTableName, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbTableName);
  _trace_stats_unique (fUnique);
  _trace_stats_accuracy (fAccuracy);
}
#endif
