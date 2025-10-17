/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 19, 2022.
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

extern int GetPrivateProfileString (LPCSTR lpszSection, LPCSTR lpszEntry,
    LPCSTR lpszDefault, LPSTR lpszRetBuffer, int cbRetBuffer,
    LPCSTR lpszFilename);

BOOL INSTAPI
SQLReadFileDSN (LPCSTR lpszFileName, LPCSTR lpszAppName, LPCSTR lpszKeyName,
    LPSTR lpszString, WORD cbString, WORD * pcbString)
{
  BOOL retcode = FALSE;
  WORD len = 0, i;
  char filename[1024];

  /* Check input parameters */
  CLEAR_ERROR ();

  if (!lpszString || !cbString)
    {
      PUSH_ERROR (ODBC_ERROR_INVALID_BUFF_LEN);
      goto quit;
    }

  if (!lpszAppName && lpszKeyName)
    {
      PUSH_ERROR (ODBC_ERROR_INVALID_REQUEST_TYPE);
      goto quit;
    }

  /* Is a file is specified */
  if (lpszFileName)
    {
      _iodbcdm_getdsnfile (lpszFileName, filename, sizeof (filename));
      len =
	  GetPrivateProfileString (lpszAppName, lpszKeyName, "", lpszString,
	  cbString, filename);
      if (numerrors == -1)
	retcode = TRUE;
      goto quit;
    }

  PUSH_ERROR (ODBC_ERROR_INVALID_PATH);
  goto quit;

quit:
  for (i = 0; i < len; i++)
    if (!lpszString[i])
      lpszString[i] = ';';

  if (pcbString)
    *pcbString = len;

  if (len == cbString - 1)
    {
      PUSH_ERROR (ODBC_ERROR_OUTPUT_STRING_TRUNCATED);
      retcode = FALSE;
    }

  return retcode;
}

BOOL INSTAPI
SQLReadFileDSNW (LPCWSTR lpszFileName, LPCWSTR lpszAppName,
    LPCWSTR lpszKeyName, LPWSTR lpszString, WORD cbString, WORD * pcbString)
{
  char *_filename_u8 = NULL;
  char *_appname_u8 = NULL;
  char *_keyname_u8 = NULL;
  char *_string_u8 = NULL;
  BOOL retcode = FALSE;

  _filename_u8 = (char *) dm_SQL_WtoU8 ((SQLWCHAR *) lpszFileName, SQL_NTS);
  if (_filename_u8 == NULL && lpszFileName)
    {
      PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
      goto done;
    }

  _appname_u8 = (char *) dm_SQL_WtoU8 ((SQLWCHAR *) lpszAppName, SQL_NTS);
  if (_appname_u8 == NULL && lpszAppName)
    {
      PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
      goto done;
    }

  _keyname_u8 = (char *) dm_SQL_WtoU8 ((SQLWCHAR *) lpszKeyName, SQL_NTS);
  if (_keyname_u8 == NULL && lpszKeyName)
    {
      PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
      goto done;
    }

  if (cbString > 0)
    {
      if ((_string_u8 = malloc (cbString * UTF8_MAX_CHAR_LEN + 1)) == NULL)
	{
	  PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
	  goto done;
	}
    }

  retcode =
      SQLReadFileDSN (_filename_u8, _appname_u8, _keyname_u8, _string_u8,
      cbString * UTF8_MAX_CHAR_LEN, pcbString);

  if (retcode == TRUE)
    {
      dm_StrCopyOut2_U8toW (_string_u8, lpszString, cbString, pcbString);
    }

done:
  MEM_FREE (_filename_u8);
  MEM_FREE (_appname_u8);
  MEM_FREE (_keyname_u8);
  MEM_FREE (_string_u8);

  return retcode;
}
