/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 3, 2024.
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
#include <iodbc.h>
#include <odbcinst.h>
#include <unicode.h>

#include "inifile.h"
#include "misc.h"
#include "iodbc_error.h"

extern BOOL ValidDSN (LPCSTR);
extern BOOL ValidDSNW (LPCSTR);

BOOL
RemoveDSNFromIni (SQLPOINTER lpszDSN, SQLCHAR waMode)
{
  BOOL retcode = FALSE;
  char *_dsn_u8 = NULL;
  PCONFIG pCfg;

  /* Check dsn */
  if (waMode == 'A')
    {
      if (!lpszDSN || !ValidDSN (lpszDSN) || !STRLEN (lpszDSN))
	{
	  PUSH_ERROR (ODBC_ERROR_INVALID_DSN);
	  goto quit;
	}
      _dsn_u8 = lpszDSN;
    }
  else
    {
      if (!lpszDSN || !ValidDSNW (lpszDSN) || !WCSLEN (lpszDSN))
	{
	  PUSH_ERROR (ODBC_ERROR_INVALID_DSN);
	  goto quit;
	}

      _dsn_u8 = (char *) dm_SQL_WtoU8 ((SQLWCHAR *) lpszDSN, SQL_NTS);
      if (_dsn_u8 == NULL && lpszDSN)
	{
	  PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
	  goto quit;
	}
    }

  if (_iodbcdm_cfg_search_init (&pCfg, "odbc.ini", TRUE))
    {
      PUSH_ERROR (ODBC_ERROR_REQUEST_FAILED);
      goto quit;
    }

  if (strcmp (_dsn_u8, "Default"))
    {
      /* deletes a DSN from [ODBC data sources] section */
#ifdef WIN32
      _iodbcdm_cfg_write (pCfg, "ODBC 32 bit Data Sources", (LPSTR) _dsn_u8,
	  NULL);
#else
      _iodbcdm_cfg_write (pCfg, "ODBC Data Sources", (LPSTR) _dsn_u8, NULL);
#endif
    }

  /* deletes the DSN section in odbc.ini */
  _iodbcdm_cfg_write (pCfg, (LPSTR) _dsn_u8, NULL, NULL);

  if (_iodbcdm_cfg_commit (pCfg))
    {
      PUSH_ERROR (ODBC_ERROR_REQUEST_FAILED);
      goto done;
    }

  retcode = TRUE;

done:
  _iodbcdm_cfg_done (pCfg);

quit:
  if (_dsn_u8 != lpszDSN)
    MEM_FREE (_dsn_u8);

  return retcode;
}


BOOL INSTAPI
SQLRemoveDSNFromIni (LPCSTR lpszDSN)
{
  BOOL retcode = FALSE;

  CLEAR_ERROR ();

  switch (configMode)
    {
    case ODBC_USER_DSN:
      wSystemDSN = USERDSN_ONLY;
      retcode = RemoveDSNFromIni ((SQLPOINTER) lpszDSN, 'A');
      goto quit;

    case ODBC_SYSTEM_DSN:
      wSystemDSN = SYSTEMDSN_ONLY;
      retcode = RemoveDSNFromIni ((SQLPOINTER) lpszDSN, 'A');
      goto quit;

    case ODBC_BOTH_DSN:
      wSystemDSN = USERDSN_ONLY;
      retcode = RemoveDSNFromIni ((SQLPOINTER) lpszDSN, 'A');
      if (!retcode)
	{
	  CLEAR_ERROR ();
	  wSystemDSN = SYSTEMDSN_ONLY;
	  retcode = RemoveDSNFromIni ((SQLPOINTER) lpszDSN, 'A');
	}
      goto quit;
    }

  PUSH_ERROR (ODBC_ERROR_GENERAL_ERR);
  goto quit;

quit:
  wSystemDSN = USERDSN_ONLY;
  configMode = ODBC_BOTH_DSN;
  return retcode;
}

BOOL INSTAPI
SQLRemoveDSNFromIniW (LPCWSTR lpszDSN)
{
  BOOL retcode = FALSE;

  CLEAR_ERROR ();

  switch (configMode)
    {
    case ODBC_USER_DSN:
      wSystemDSN = USERDSN_ONLY;
      retcode = RemoveDSNFromIni ((SQLPOINTER) lpszDSN, 'W');
      goto quit;

    case ODBC_SYSTEM_DSN:
      wSystemDSN = SYSTEMDSN_ONLY;
      retcode = RemoveDSNFromIni ((SQLPOINTER) lpszDSN, 'W');
      goto quit;

    case ODBC_BOTH_DSN:
      wSystemDSN = USERDSN_ONLY;
      retcode = RemoveDSNFromIni ((SQLPOINTER) lpszDSN, 'W');
      if (!retcode)
	{
	  CLEAR_ERROR ();
	  wSystemDSN = SYSTEMDSN_ONLY;
	  retcode = RemoveDSNFromIni ((SQLPOINTER) lpszDSN, 'W');
	}
      goto quit;
    }

  PUSH_ERROR (ODBC_ERROR_GENERAL_ERR);
  goto quit;

quit:
  wSystemDSN = USERDSN_ONLY;
  configMode = ODBC_BOTH_DSN;
  return retcode;
}
