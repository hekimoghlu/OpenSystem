/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 5, 2022.
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

extern BOOL ValidDSN (LPCSTR lpszDSN);
extern BOOL ValidDSNW (LPCWSTR lpszDSN);

extern int GetPrivateProfileString (LPCSTR lpszSection, LPCSTR lpszEntry,
    LPCSTR lpszDefault, LPSTR lpszRetBuffer, int cbRetBuffer,
    LPCSTR lpszFilename);

BOOL
WriteDSNToIni (LPCSTR lpszDSN, LPCSTR lpszDriver)
{
  char szBuffer[4096];
  BOOL retcode = FALSE;
  PCONFIG pCfg = NULL;

  if (_iodbcdm_cfg_search_init (&pCfg, "odbc.ini", TRUE))
    {
      PUSH_ERROR (ODBC_ERROR_REQUEST_FAILED);
      goto done;
    }

  if (strcmp (lpszDSN, "Default"))
    {
      /* adds a DSN=Driver to the [ODBC data sources] section */
#ifdef WIN32
      if (_iodbcdm_cfg_write (pCfg, "ODBC 32 bit Data Sources",
	      (LPSTR) lpszDSN, (LPSTR) lpszDriver))
#else
      if (_iodbcdm_cfg_write (pCfg, "ODBC Data Sources", (LPSTR) lpszDSN,
	      (LPSTR) lpszDriver))
#endif
	{
	  PUSH_ERROR (ODBC_ERROR_REQUEST_FAILED);
	  goto done;
	}
    }

  /* deletes the DSN section in odbc.ini */
  if (_iodbcdm_cfg_write (pCfg, (LPSTR) lpszDSN, NULL, NULL))
    {
      PUSH_ERROR (ODBC_ERROR_REQUEST_FAILED);
      goto done;
    }

  /* gets the file of the driver if lpszDriver is a valid description */
  wSystemDSN = USERDSN_ONLY;
  if (!GetPrivateProfileString ((LPSTR) lpszDriver, "Driver", "", szBuffer,
	  sizeof (szBuffer) - 1, "odbcinst.ini"))
    {
      wSystemDSN = SYSTEMDSN_ONLY;

      if (!GetPrivateProfileString ((LPSTR) lpszDriver, "Driver", "",
	      szBuffer, sizeof (szBuffer) - 1, "odbcinst.ini"))
	{
	  PUSH_ERROR (ODBC_ERROR_INVALID_NAME);
	  goto done;
	}
    }

  /* adds a [DSN] section with Driver key */
  if (_iodbcdm_cfg_write (pCfg, (LPSTR) lpszDSN, "Driver", szBuffer)
      || _iodbcdm_cfg_commit (pCfg))
    {
      PUSH_ERROR (ODBC_ERROR_REQUEST_FAILED);
      goto done;
    }

  retcode = TRUE;

done:
  wSystemDSN = USERDSN_ONLY;
  configMode = ODBC_BOTH_DSN;
  if (pCfg)
    _iodbcdm_cfg_done (pCfg);
  return retcode;
}


BOOL INSTAPI
SQLWriteDSNToIni_Internal (SQLPOINTER lpszDSN, SQLPOINTER lpszDriver,
    SQLCHAR waMode)
{
  char *_driver_u8 = NULL;
  char *_dsn_u8 = NULL;
  BOOL retcode = FALSE;

  /* Check input parameters */
  CLEAR_ERROR ();

  if (waMode == 'A')
    {
      if (!lpszDSN || !ValidDSN (lpszDSN) || !STRLEN (lpszDSN))
	{
	  PUSH_ERROR (ODBC_ERROR_INVALID_DSN);
	  goto quit;
	}
    }
  else
    {
      if (!lpszDSN || !ValidDSNW (lpszDSN) || !WCSLEN (lpszDSN))
	{
	  PUSH_ERROR (ODBC_ERROR_INVALID_DSN);
	  goto quit;
	}
    }

  if (waMode == 'W')
    {
      _dsn_u8 = (char *) dm_SQL_WtoU8 ((SQLWCHAR *) lpszDSN, SQL_NTS);
      if (_dsn_u8 == NULL && lpszDSN)
	{
	  PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
	  goto quit;
	}
    }
  else
    _dsn_u8 = (SQLCHAR *) lpszDSN;

  if (waMode == 'W')
    {
      _driver_u8 = (char *) dm_SQL_WtoU8 ((SQLWCHAR *) lpszDriver, SQL_NTS);
      if (_driver_u8 == NULL && lpszDriver)
	{
	  PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
	  goto quit;
	}
    }
  else
    _driver_u8 = (SQLCHAR *) lpszDriver;

  if (!_driver_u8 || !STRLEN (_driver_u8))
    {
      PUSH_ERROR (ODBC_ERROR_INVALID_NAME);
      goto quit;
    }

  switch (configMode)
    {
    case ODBC_USER_DSN:
      wSystemDSN = USERDSN_ONLY;
      retcode = WriteDSNToIni (_dsn_u8, _driver_u8);
      goto quit;

    case ODBC_SYSTEM_DSN:
      wSystemDSN = SYSTEMDSN_ONLY;
      retcode = WriteDSNToIni (_dsn_u8, _driver_u8);
      goto quit;

    case ODBC_BOTH_DSN:
      wSystemDSN = USERDSN_ONLY;
      retcode = WriteDSNToIni (_dsn_u8, _driver_u8);
      if (!retcode)
	{
	  CLEAR_ERROR ();
	  wSystemDSN = SYSTEMDSN_ONLY;
	  retcode = WriteDSNToIni (_dsn_u8, _driver_u8);
	}
      goto quit;
    }

  PUSH_ERROR (ODBC_ERROR_GENERAL_ERR);
  goto quit;

quit:
  if (_dsn_u8 != lpszDSN)
    MEM_FREE (_dsn_u8);
  if (_driver_u8 != lpszDriver)
    MEM_FREE (_driver_u8);

  wSystemDSN = USERDSN_ONLY;
  configMode = ODBC_BOTH_DSN;

  return retcode;
}

BOOL INSTAPI
SQLWriteDSNToIni (LPCSTR lpszDSN, LPCSTR lpszDriver)
{
  return SQLWriteDSNToIni_Internal ((SQLPOINTER) lpszDSN,
      (SQLPOINTER) lpszDriver, 'A');
}

BOOL INSTAPI
SQLWriteDSNToIniW (LPCWSTR lpszDSN, LPCWSTR lpszDriver)
{
  return SQLWriteDSNToIni_Internal ((SQLPOINTER) lpszDSN,
      (SQLPOINTER) lpszDriver, 'W');
}
