/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 13, 2022.
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

static void
_trace_func_name (SQLUSMALLINT fFunc, int format)
{
  char *ptr = "unknown function";

  switch (fFunc)
    {
/* All ODBC 2.x functions */
      _S (SQL_API_ALL_FUNCTIONS);

/* ODBC 2.x */
      _S (SQL_API_SQLALLOCCONNECT);
      _S (SQL_API_SQLALLOCENV);
      _S (SQL_API_SQLALLOCSTMT);
      _S (SQL_API_SQLBINDCOL);
      _S (SQL_API_SQLBINDPARAMETER);
      _S (SQL_API_SQLBROWSECONNECT);
      _S (SQL_API_SQLCANCEL);
#if (ODBCVER < 0x0300)
      _S (SQL_API_SQLCOLATTRIBUTES);
#endif
      _S (SQL_API_SQLCOLUMNPRIVILEGES);
      _S (SQL_API_SQLCOLUMNS);
      _S (SQL_API_SQLCONNECT);
      _S (SQL_API_SQLDATASOURCES);
      _S (SQL_API_SQLDESCRIBECOL);
      _S (SQL_API_SQLDESCRIBEPARAM);
      _S (SQL_API_SQLDISCONNECT);
      _S (SQL_API_SQLDRIVERCONNECT);
      _S (SQL_API_SQLDRIVERS);
      _S (SQL_API_SQLERROR);
      _S (SQL_API_SQLEXECDIRECT);
      _S (SQL_API_SQLEXECUTE);
      _S (SQL_API_SQLEXTENDEDFETCH);
      _S (SQL_API_SQLFETCH);
      _S (SQL_API_SQLFOREIGNKEYS);
      _S (SQL_API_SQLFREECONNECT);
      _S (SQL_API_SQLFREEENV);
      _S (SQL_API_SQLFREESTMT);
      _S (SQL_API_SQLGETCONNECTOPTION);
      _S (SQL_API_SQLGETCURSORNAME);
      _S (SQL_API_SQLGETDATA);
      _S (SQL_API_SQLGETFUNCTIONS);
      _S (SQL_API_SQLGETINFO);
      _S (SQL_API_SQLGETSTMTOPTION);
      _S (SQL_API_SQLGETTYPEINFO);
      _S (SQL_API_SQLMORERESULTS);
      _S (SQL_API_SQLNATIVESQL);
      _S (SQL_API_SQLNUMPARAMS);
      _S (SQL_API_SQLNUMRESULTCOLS);
      _S (SQL_API_SQLPARAMDATA);
      _S (SQL_API_SQLPARAMOPTIONS);
      _S (SQL_API_SQLPREPARE);
      _S (SQL_API_SQLPRIMARYKEYS);
      _S (SQL_API_SQLPROCEDURECOLUMNS);
      _S (SQL_API_SQLPROCEDURES);
      _S (SQL_API_SQLPUTDATA);
      _S (SQL_API_SQLROWCOUNT);
      _S (SQL_API_SQLSETCONNECTOPTION);
      _S (SQL_API_SQLSETCURSORNAME);
      _S (SQL_API_SQLSETPARAM);
      _S (SQL_API_SQLSETPOS);
      _S (SQL_API_SQLSETSCROLLOPTIONS);
      _S (SQL_API_SQLSETSTMTOPTION);
      _S (SQL_API_SQLSPECIALCOLUMNS);
      _S (SQL_API_SQLSTATISTICS);
      _S (SQL_API_SQLTABLEPRIVILEGES);
      _S (SQL_API_SQLTABLES);
      _S (SQL_API_SQLTRANSACT);
#if (ODBCVER >= 0x0300)
/* All ODBC 2.x functions */
      _S (SQL_API_ODBC3_ALL_FUNCTIONS);

/* ODBC 3.x */
      _S (SQL_API_SQLALLOCHANDLE);
      _S (SQL_API_SQLALLOCHANDLESTD);
      _S (SQL_API_SQLBINDPARAM);
      _S (SQL_API_SQLBULKOPERATIONS);
      _S (SQL_API_SQLCLOSECURSOR);
      _S (SQL_API_SQLCOLATTRIBUTE);
      _S (SQL_API_SQLCOPYDESC);
      _S (SQL_API_SQLENDTRAN);
      _S (SQL_API_SQLFETCHSCROLL);
      _S (SQL_API_SQLFREEHANDLE);
      _S (SQL_API_SQLGETCONNECTATTR);
      _S (SQL_API_SQLGETDESCFIELD);
      _S (SQL_API_SQLGETDESCREC);
      _S (SQL_API_SQLGETDIAGFIELD);
      _S (SQL_API_SQLGETDIAGREC);
      _S (SQL_API_SQLGETENVATTR);
      _S (SQL_API_SQLGETSTMTATTR);
      _S (SQL_API_SQLSETCONNECTATTR);
      _S (SQL_API_SQLSETDESCFIELD);
      _S (SQL_API_SQLSETDESCREC);
      _S (SQL_API_SQLSETENVATTR);
      _S (SQL_API_SQLSETSTMTATTR);

#endif
    }

  if (format)
    trace_emit ("\t\t%-15.15s   %d (%s)\n", "SQLUSMALLINT", (int) fFunc, ptr);
  else
    trace_emit_string (ptr, SQL_NTS, 0);
}


void
_trace_func_result (
    SQLUSMALLINT	  fFunc, 
    SQLUSMALLINT	* pfExists, 
    int			  output)
{
  int i;

  if (fFunc == SQL_API_ALL_FUNCTIONS)
    {
      _trace_usmallint_p (pfExists, 0);

      if (!output)
        return;

      for (i = 1; i < 100; i++)
	if (pfExists[i])
	  _trace_func_name (i, 0);
    }
#if (ODBCVER >= 0x0300)
  else if (fFunc == SQL_API_ODBC3_ALL_FUNCTIONS)
    {
      if (!output)
        return;

      _trace_usmallint_p (pfExists, 0);
      for (i = 1; i < SQL_API_ODBC3_ALL_FUNCTIONS; i++)
	if (SQL_FUNC_EXISTS (pfExists, i))
	  _trace_func_name (i, 0);
    }
#endif
  else
    {
      _trace_usmallint_p (pfExists, output);
    }
}


void
trace_SQLGetFunctions (int trace_leave, int retcode,
  SQLHDBC		  hdbc,
  SQLUSMALLINT		  fFunc,
  SQLUSMALLINT     	* pfExists)
{
  /* Trace function */
  _trace_print_function (en_GetFunctions, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_DBC, hdbc);
  _trace_func_name (fFunc, 1);
  _trace_func_result (fFunc, pfExists, trace_leave);
}
