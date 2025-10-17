/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 20, 2024.
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


/*
 *  Never print plaintext passwords
 *
 *  NOTE: This function modifies the original string
 *  
 */
static void
_trace_connstr_hidepwd (SQLCHAR *str)
{
  SQLCHAR *ptr;
  int state = 0;

  for (ptr = str; *ptr;)
    {
      switch (state)
	{
	case -1:
	  if (strchr ("\'\"}", *ptr))
	    state = 0;
	  break;

	case 0:
	  if (toupper(*ptr) == 'P')
	    state = 1;
	  else if (strchr ("\'\"{", *ptr))
	    state = -1;		/* in string */
	  break;

	case 1:
	  if (toupper(*ptr) == 'W')
	    state = 2;
	  else
	    state = 0;
	  break;

	case 2:
	  if (toupper(*ptr) == 'D')
	    state = 3;
	  else
	    state = 0;
	  break;

	case 3:
	  if (*ptr == '=')
	    state = 4;		/* goto password mode */
	  else
	    state = 0;
	  break;

	case 4:
	  if (*ptr == ';')
	    {
	      state = 0;	/* go back to normal mode */
	    }
	  else
	    *ptr = '*';
	  break;
	}
      ptr++;
    }
}


static void
_trace_connstr (
  SQLCHAR		* str, 
  SQLSMALLINT		  len, 
  SQLSMALLINT		* lenptr, 
  int 			  output)
{
  SQLCHAR *dup;
  ssize_t length;

  if (!str)
    {
      trace_emit ("\t\t%-15.15s * 0x0\n", "SQLCHAR");
      return;
    }

  trace_emit ("\t\t%-15.15s * %p\n", "SQLCHAR", str);

  if (!output)
    return;

  /*
   *  Calculate string length
   */
  if (lenptr )
    length = *lenptr;
  else
    length = len;

  if (length == SQL_NTS)
    length = STRLEN (str);


  /*
   *  Make a copy of the string
   */
  if ((dup = (SQLCHAR *) malloc (length + 1)) == NULL)
    return;
  memcpy (dup, str, length);
  dup[length] = '\0';

  /*
   *  Emit the string
   */
  _trace_connstr_hidepwd (dup);
  trace_emit_string (dup, length, 0);
  free (dup);
}


static void
_trace_connstr_w (
  SQLWCHAR		* str, 
  SQLSMALLINT		  len, 
  SQLSMALLINT		* lenptr, 
  int 			  output)
{
  SQLCHAR *dup;
  long length;

  if (!str)
    {
      trace_emit ("\t\t%-15.15s * 0x0\n", "SQLWCHAR");
      return;
    }

  trace_emit ("\t\t%-15.15s * %p\n", "SQLWCHAR", str);

  if (!output)
    return;

  /*
   *  Calculate string length
   */
  if (lenptr)
    length = *lenptr;
  else
    length = len;

  /* 
   * Emit the string
   */
  dup = dm_SQL_W2A (str, length);
  _trace_connstr_hidepwd (dup);
  trace_emit_string (dup, SQL_NTS, 1);
  free (dup);
}



static void
_trace_drvcn_completion(SQLUSMALLINT fDriverCompletion)
{
  char *ptr = "invalid completion value";

  switch (fDriverCompletion)
    {
      _S (SQL_DRIVER_PROMPT);
      _S (SQL_DRIVER_COMPLETE);
      _S (SQL_DRIVER_COMPLETE_REQUIRED);
      _S (SQL_DRIVER_NOPROMPT);
    }

  trace_emit ("\t\t%-15.15s   %d (%s)\n", 
  	"SQLUSMALLINT", (int) fDriverCompletion, ptr);
}


void 
trace_SQLDriverConnect (int trace_leave, int retcode,
  SQLHDBC		  hdbc,
  SQLHWND		  hwnd,
  SQLCHAR		* szConnStrIn,
  SQLSMALLINT		  cbConnStrIn,
  SQLCHAR 		* szConnStrOut,
  SQLSMALLINT		  cbConnStrOutMax,
  SQLSMALLINT 	 	* pcbConnStrOut,
  SQLUSMALLINT		  fDriverCompletion)
{
  /* Trace function */
  _trace_print_function (en_DriverConnect, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_DBC, hdbc);
  _trace_pointer (hwnd);
  _trace_connstr (szConnStrIn, cbConnStrIn, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbConnStrIn);
  _trace_connstr (szConnStrOut, cbConnStrOutMax, pcbConnStrOut,
      TRACE_OUTPUT_SUCCESS);
  _trace_stringlen ("SQLSMALLINT", cbConnStrOutMax);
  _trace_smallint_p (pcbConnStrOut, TRACE_OUTPUT_SUCCESS);
  _trace_drvcn_completion (fDriverCompletion);
}


#if ODBCVER >= 0x0300
void 
trace_SQLDriverConnectW (int trace_leave, int retcode,
  SQLHDBC		  hdbc,
  SQLHWND		  hwnd,
  SQLWCHAR 		* szConnStrIn,
  SQLSMALLINT		  cbConnStrIn,
  SQLWCHAR 		* szConnStrOut,
  SQLSMALLINT		  cbConnStrOutMax,
  SQLSMALLINT 	 	* pcbConnStrOut,
  SQLUSMALLINT		  fDriverCompletion)
{
  /* Trace function */
  _trace_print_function (en_DriverConnectW, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_DBC, hdbc);
  _trace_pointer (hwnd);
  _trace_connstr_w (szConnStrIn, cbConnStrIn, NULL, TRACE_INPUT);
  _trace_stringlen ("SQLSMALLINT", cbConnStrIn);
  _trace_connstr_w (szConnStrOut, cbConnStrOutMax, pcbConnStrOut,
      TRACE_OUTPUT_SUCCESS);
  _trace_stringlen ("SQLSMALLINT", cbConnStrOutMax);
  _trace_smallint_p (pcbConnStrOut, TRACE_OUTPUT_SUCCESS);
  _trace_drvcn_completion (fDriverCompletion);
}
#endif
