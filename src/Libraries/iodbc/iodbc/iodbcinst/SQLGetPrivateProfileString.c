/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 23, 2025.
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
int
GetPrivateProfileString (LPCSTR lpszSection, LPCSTR lpszEntry,
    LPCSTR lpszDefault, LPSTR lpszRetBuffer, int cbRetBuffer,
    LPCSTR lpszFilename)
{
  char *defval = (char *) lpszDefault, *value = NULL;
  int len = 0;
  PCONFIG pCfg;

  lpszRetBuffer[0] = 0;

  /* If error during reading the file */
  if (_iodbcdm_cfg_search_init (&pCfg, lpszFilename, FALSE))
    {
      if (lpszDefault)
	STRNCPY (lpszRetBuffer, lpszDefault, cbRetBuffer - 1);
      PUSH_ERROR (ODBC_ERROR_INVALID_PATH);
      goto fail;
    }

  /* List all sections from the ini file */
  if (lpszSection == NULL || *lpszSection == '\0')
    {
      len = _iodbcdm_list_sections (pCfg, lpszRetBuffer, cbRetBuffer);
      goto done;
    }

  /* List all the entries of the specified section */
  if (lpszEntry == NULL || *lpszEntry == '\0')
    {
      len =
	  _iodbcdm_list_entries (pCfg, lpszSection, lpszRetBuffer,
	  cbRetBuffer);
      goto done;
    }

  /*
   *  Sorry for this one -- Windows cannot handle a default value of
   *  "" in GetPrivateProfileString, so it is passed as " " instead.
   */
  if (defval == NULL || *defval == '\0')
    defval = " ";

  /*
   *  Check whether someone else has modified the odbc.ini file
   */
  _iodbcdm_cfg_refresh (pCfg);

  if (!_iodbcdm_cfg_find (pCfg, (LPSTR) lpszSection, (LPSTR) lpszEntry))
    value = pCfg->value;

  if (value == NULL)
    {
      value = defval;

      if (value[0] == ' ' && value[1] == '\0')
	value = "";
    }

  STRNCPY (lpszRetBuffer, value, cbRetBuffer - 1);

done:
  _iodbcdm_cfg_done (pCfg);

fail:
  if (!len)
    len = STRLEN (lpszRetBuffer);

  if (len == cbRetBuffer - 1)
    PUSH_ERROR (ODBC_ERROR_INVALID_BUFF_LEN);

  return len;
}
#endif


int INSTAPI
SQLGetPrivateProfileString (LPCSTR lpszSection, LPCSTR lpszEntry,
    LPCSTR lpszDefault, LPSTR lpszRetBuffer, int cbRetBuffer,
    LPCSTR lpszFilename)
{
  char pathbuf[1024];
  int len = 0;

  /* Check input parameters */
  CLEAR_ERROR ();

  if (!lpszRetBuffer || !cbRetBuffer)
    {
      PUSH_ERROR (ODBC_ERROR_INVALID_BUFF_LEN);
      goto quit;
    }

  if (!lpszDefault)
    {
      PUSH_ERROR (ODBC_ERROR_GENERAL_ERR);
      goto quit;
    }

  /* Else go through user/system odbc.ini */
  switch (configMode)
    {
    case ODBC_USER_DSN:
      wSystemDSN = USERDSN_ONLY;
      if (lpszFilename)
	{
	  len =
	      GetPrivateProfileString (lpszSection, lpszEntry, lpszDefault,
	      lpszRetBuffer, cbRetBuffer, lpszFilename);
	  goto quit;
	}
      if (_iodbcadm_getinifile (pathbuf, sizeof (pathbuf), FALSE, FALSE))
	len =
	    GetPrivateProfileString (lpszSection, lpszEntry, lpszDefault,
	    lpszRetBuffer, cbRetBuffer, pathbuf);
      goto quit;

    case ODBC_SYSTEM_DSN:
      wSystemDSN = SYSTEMDSN_ONLY;
      if (lpszFilename)
	{
	  len =
	      GetPrivateProfileString (lpszSection, lpszEntry, lpszDefault,
	      lpszRetBuffer, cbRetBuffer, lpszFilename);
	  goto quit;
	}
      if (_iodbcadm_getinifile (pathbuf, sizeof (pathbuf), FALSE, FALSE))
	len =
	    GetPrivateProfileString (lpszSection, lpszEntry, lpszDefault,
	    lpszRetBuffer, cbRetBuffer, pathbuf);
      goto quit;

    case ODBC_BOTH_DSN:
      wSystemDSN = USERDSN_ONLY;
      if (lpszFilename)
	{
	  len =
	      GetPrivateProfileString (lpszSection, lpszEntry, lpszDefault,
	      lpszRetBuffer, cbRetBuffer, lpszFilename);
	  if (!len)
	    {
	      CLEAR_ERROR ();
	      wSystemDSN = SYSTEMDSN_ONLY;
	      len =
		  GetPrivateProfileString (lpszSection, lpszEntry,
		  lpszDefault, lpszRetBuffer, cbRetBuffer, lpszFilename);
	    }
	  goto quit;
	}
      if (_iodbcadm_getinifile (pathbuf, sizeof (pathbuf), FALSE, FALSE))
        {
	  len =
	    GetPrivateProfileString (lpszSection, lpszEntry, lpszDefault,
	    lpszRetBuffer, cbRetBuffer, pathbuf);
          if (len)
            goto quit;
        }
      CLEAR_ERROR ();
      wSystemDSN = SYSTEMDSN_ONLY;
      if (_iodbcadm_getinifile (pathbuf, sizeof (pathbuf), FALSE, FALSE))
        {
          len =
  	    GetPrivateProfileString (lpszSection, lpszEntry, lpszDefault,
		lpszRetBuffer, cbRetBuffer, pathbuf);
	}
      goto quit;
    }

  PUSH_ERROR (ODBC_ERROR_GENERAL_ERR);
  goto quit;

quit:
  wSystemDSN = USERDSN_ONLY;
  configMode = ODBC_BOTH_DSN;
  return len;
}

int INSTAPI
SQLGetPrivateProfileStringW (LPCWSTR lpszSection, LPCWSTR lpszEntry,
    LPCWSTR lpszDefault, LPWSTR lpszRetBuffer, int cbRetBuffer,
    LPCWSTR lpszFilename)
{
  char *_section_u8 = NULL;
  char *_entry_u8 = NULL;
  char *_default_u8 = NULL;
  char *_buffer_u8 = NULL;
  char *_filename_u8 = NULL;
  SQLCHAR *ptr;
  SQLWCHAR *ptrW;
  SQLSMALLINT length, len;

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

  _default_u8 = (char *) dm_SQL_WtoU8 ((SQLWCHAR *) lpszDefault, SQL_NTS);
  if (_default_u8 == NULL && lpszDefault)
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

  if (cbRetBuffer > 0)
    {
      if ((_buffer_u8 = malloc (cbRetBuffer * UTF8_MAX_CHAR_LEN + 1)) == NULL)
	{
	  PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
	  goto done;
	}
    }

  length = SQLGetPrivateProfileString (_section_u8, _entry_u8, _default_u8,
      _buffer_u8, cbRetBuffer * UTF8_MAX_CHAR_LEN, _filename_u8);

  if (length > 0)
    {
      if (lpszSection == NULL || lpszEntry == NULL ||
	  lpszSection[0] == '\0' || lpszEntry[0] == '\0')
	{
	  length = 0;

	  for (ptr = _buffer_u8, ptrW = lpszRetBuffer; *ptr;
	      ptr += STRLEN (ptr) + 1, ptrW += WCSLEN (ptrW) + 1)
	    {
	      dm_StrCopyOut2_U8toW (ptr, ptrW, cbRetBuffer - length - 1,
		  &len);
	      length += len;
	    }

	  *ptrW = L'\0';
	  length++;
	}
      else
	{
	  dm_StrCopyOut2_U8toW (_buffer_u8, lpszRetBuffer, cbRetBuffer,
	      &length);
	}
    }
  else
    {
      dm_StrCopyOut2_U8toW (_buffer_u8, lpszRetBuffer, cbRetBuffer, &length);
    }

done:
  MEM_FREE (_section_u8);
  MEM_FREE (_entry_u8);
  MEM_FREE (_default_u8);
  MEM_FREE (_buffer_u8);
  MEM_FREE (_filename_u8);

  return length;
}

BOOL INSTAPI
SQLGetKeywordValue (LPCSTR lpszSection,
    LPCSTR lpszEntry, LPSTR lpszBuffer, int cbBuffer, int *pcbBufOut)
{
  int ret =
      SQLGetPrivateProfileString (lpszSection, lpszEntry, "", lpszBuffer,
      cbBuffer, "odbc.ini");

  if (pcbBufOut)
    *pcbBufOut = ret;

  return (ret != 0);
}

BOOL INSTAPI
SQLGetKeywordValueW (LPCWSTR lpszSection,
    LPCWSTR lpszEntry, LPWSTR lpszBuffer, int cbBuffer, int *pcbBufOut)
{
  int ret =
      SQLGetPrivateProfileStringW (lpszSection, lpszEntry, L"", lpszBuffer,
      cbBuffer, L"odbc.ini");

  if (pcbBufOut)
    *pcbBufOut = ret;

  return (ret != 0);
}
