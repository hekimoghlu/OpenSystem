/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 3, 2024.
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
#include "iodbc_error.h"
#include "misc.h"

#ifndef WIN32
BOOL
WritePrivateProfileString (LPCSTR lpszSection, LPCSTR lpszEntry,
    LPCSTR lpszString, LPCSTR lpszFilename)
{
  BOOL retcode = FALSE;
  PCONFIG pCfg = NULL;

  /* Check Input parameters */
  if (lpszSection == NULL || *lpszSection == '\0')
    {
      PUSH_ERROR (ODBC_ERROR_INVALID_REQUEST_TYPE);
      goto fail;
    }

  /* If error during reading the file */
  if (_iodbcdm_cfg_search_init (&pCfg, lpszFilename, TRUE))
    {
      PUSH_ERROR (ODBC_ERROR_GENERAL_ERR);
      goto fail;
    }

  /* Check if the section must be deleted */
  if (!lpszEntry)
    {
      _iodbcdm_cfg_write (pCfg, (LPSTR) lpszSection, NULL, NULL);
      goto done;
    }

  /* Check if the entry must be deleted */
  if (!lpszString)
    {
      _iodbcdm_cfg_write (pCfg, (LPSTR) lpszSection, (LPSTR) lpszEntry, NULL);
      goto done;
    }

  /* Else add the entry */
  _iodbcdm_cfg_write (pCfg, (LPSTR) lpszSection, (LPSTR) lpszEntry,
      (LPSTR) lpszString);

done:
  if (!_iodbcdm_cfg_commit (pCfg))
    retcode = TRUE;
  else
    {
      PUSH_ERROR (ODBC_ERROR_REQUEST_FAILED);
      goto fail;
    }

fail:
  if (pCfg)
    _iodbcdm_cfg_done (pCfg);
  return retcode;
}
#endif


BOOL INSTAPI
SQLWritePrivateProfileString (LPCSTR lpszSection, LPCSTR lpszEntry,
    LPCSTR lpszString, LPCSTR lpszFilename)
{
  char pathbuf[1024];
  BOOL retcode = FALSE;

  /* Check input parameters */
  CLEAR_ERROR ();

  /* Else go through user/system odbc.ini */
  switch (configMode)
    {
    case ODBC_USER_DSN:
      wSystemDSN = USERDSN_ONLY;
      if (lpszFilename)
	{
	  retcode =
	      WritePrivateProfileString (lpszSection, lpszEntry, lpszString,
	      lpszFilename);
	  goto quit;
	}
      if (_iodbcadm_getinifile (pathbuf, sizeof (pathbuf), FALSE, TRUE))
	retcode =
	    WritePrivateProfileString (lpszSection, lpszEntry, lpszString,
	    pathbuf);
      goto quit;

    case ODBC_SYSTEM_DSN:
      wSystemDSN = SYSTEMDSN_ONLY;
      if (lpszFilename)
	{
	  retcode =
	      WritePrivateProfileString (lpszSection, lpszEntry, lpszString,
	      lpszFilename);
	  goto quit;
	}
      if (_iodbcadm_getinifile (pathbuf, sizeof (pathbuf), FALSE, TRUE))
	retcode =
	    WritePrivateProfileString (lpszSection, lpszEntry, lpszString,
	    pathbuf);
      goto quit;

    case ODBC_BOTH_DSN:
      wSystemDSN = USERDSN_ONLY;
      if (lpszFilename)
	{
	  retcode =
	      WritePrivateProfileString (lpszSection, lpszEntry, lpszString,
	      lpszFilename);
	  if (!retcode)
	    {
	      CLEAR_ERROR ();
	      wSystemDSN = SYSTEMDSN_ONLY;
	      retcode =
		  WritePrivateProfileString (lpszSection, lpszEntry,
		  lpszString, lpszFilename);
	    }
	  goto quit;
	}
      if (_iodbcadm_getinifile (pathbuf, sizeof (pathbuf), FALSE, TRUE))
	retcode =
	    WritePrivateProfileString (lpszSection, lpszEntry, lpszString,
	    pathbuf);
      else
	{
	  CLEAR_ERROR ();
	  wSystemDSN = SYSTEMDSN_ONLY;
	  if (_iodbcadm_getinifile (pathbuf, sizeof (pathbuf), FALSE, TRUE))
	    retcode =
		WritePrivateProfileString (lpszSection, lpszEntry, lpszString,
		pathbuf);
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
SQLWritePrivateProfileStringW (LPCWSTR lpszSection, LPCWSTR lpszEntry,
    LPCWSTR lpszString, LPCWSTR lpszFilename)
{
  char *_section_u8 = NULL;
  char *_entry_u8 = NULL;
  char *_string_u8 = NULL;
  char *_filename_u8 = NULL;
  BOOL retcode = FALSE;

  _section_u8 = (char *) dm_SQL_WtoU8 ((SQLWCHAR *) lpszSection, SQL_NTS);
  if (_section_u8 == NULL && lpszSection)
    {
      PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
      goto done;
    }

  _entry_u8 = (char *) dm_SQL_WtoU8 ((SQLWCHAR *) lpszEntry, SQL_NTS);
  if (_entry_u8 == NULL && lpszEntry)
    {
      PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
      goto done;
    }

  _string_u8 = (char *) dm_SQL_WtoU8 ((SQLWCHAR *) lpszString, SQL_NTS);
  if (_string_u8 == NULL && lpszString)
    {
      PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
      goto done;
    }

  _filename_u8 = (char *) dm_SQL_WtoU8 ((SQLWCHAR *) lpszFilename, SQL_NTS);
  if (_filename_u8 == NULL && lpszFilename)
    {
      PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
      goto done;
    }

  retcode =
      SQLWritePrivateProfileString (_section_u8, _entry_u8, _string_u8,
      _filename_u8);

done:
  MEM_FREE (_section_u8);
  MEM_FREE (_entry_u8);
  MEM_FREE (_string_u8);
  MEM_FREE (_filename_u8);

  return retcode;
}

BOOL INSTAPI
SQLSetKeywordValue (LPCSTR lpszSection,
    LPCSTR lpszEntry, LPSTR lpszString, int cbString)
{
  return SQLWritePrivateProfileString (lpszSection,
      lpszEntry, lpszString, "odbc.ini");
}

BOOL INSTAPI
SQLSetKeywordValueW (LPCWSTR lpszSection,
    LPCWSTR lpszEntry, LPWSTR lpszString, int cbString)
{
  return SQLWritePrivateProfileStringW (lpszSection,
      lpszEntry, lpszString, L"odbc.ini");
}
