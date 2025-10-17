/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 21, 2023.
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
#include "gui.h"

#include <iodbc.h>
#include <herr.h>
#include <dlproc.h>


SQLRETURN SQL_API
_iodbcdm_drvconn_dialbox (
    HWND	  hwnd,
    LPSTR	  szInOutConnStr,
    DWORD	  cbInOutConnStr,
    int	 	* sqlStat,
    SQLUSMALLINT  fDriverCompletion,
    UWORD	* config)
{
  RETCODE retcode = SQL_ERROR;
  char *szDSN = NULL, *szDriver = NULL, *szUID = NULL, *szPWD = NULL, *curr;
  TLOGIN log_t;

  /* Check input parameters */
  if (!hwnd || !szInOutConnStr || cbInOutConnStr < 1)
    goto quit;

  /* Check if the DSN is already set or DRIVER */
  for (curr = szInOutConnStr; *curr; curr += (STRLEN (curr) + 1))
    {
      if (!strncasecmp (curr, "DSN=", STRLEN ("DSN=")))
	{
	  szDSN = curr + STRLEN ("DSN=");
	  continue;
	}
      if (!strncasecmp (curr, "DRIVER=", STRLEN ("DRIVER=")))
	{
	  szDriver = curr + STRLEN ("DRIVER=");
	  continue;
	}
      if (!strncasecmp (curr, "UID=", STRLEN ("UID=")))
	{
	  szUID = curr + STRLEN ("UID=");
	  continue;
	}
      if (!strncasecmp (curr, "UserName=", STRLEN ("UserName=")))
	{
	  szUID = curr + STRLEN ("UserName=");
	  continue;
	}
      if (!strncasecmp (curr, "LastUser=", STRLEN ("LastUser=")))
	{
	  szUID = curr + STRLEN ("LastUser=");
	  continue;
	}
      if (!strncasecmp (curr, "PWD=", STRLEN ("PWD=")))
	{
	  szPWD = curr + STRLEN ("PWD=");
	  continue;
	}
      if (!strncasecmp (curr, "Password=", STRLEN ("Password=")))
	{
	  szPWD = curr + STRLEN ("Password=");
	  continue;
	}
    }

  if (fDriverCompletion != SQL_DRIVER_NOPROMPT && (!szUID || !szPWD))
    {
      create_login (hwnd, szUID, szPWD, szDSN ? szDSN : "(File DSN)", &log_t);

      if (log_t.user && !szUID)
	{
	  sprintf (curr, "UID=%s", log_t.user);
	  curr += STRLEN (curr);
	  *curr++ = '\0';
	  free (log_t.user);
	}

      if (log_t.pwd && !szPWD)
	{
	  sprintf (curr, "PWD=%s", log_t.pwd);
	  curr += STRLEN (curr);
	  *curr++ = '\0';
	  free (log_t.pwd);
	}

      /* add list-terminating '\0' */
      *curr = '\0';
    }

  retcode = log_t.ok ? SQL_SUCCESS : SQL_NO_DATA_FOUND;

quit:
  for (curr = szInOutConnStr; *curr; curr = szDSN + 1)
    {
      szDSN = curr + STRLEN (curr);
      if (szDSN[1])
	szDSN[0] = ';';
    }

  return retcode;
}
