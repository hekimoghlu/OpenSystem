/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 2, 2024.
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

extern BOOL WritePrivateProfileString (LPCSTR lpszSection, LPCSTR lpszEntry,
    LPCSTR lpszString, LPCSTR lpszFilename);

BOOL INSTAPI
SQLWriteFileDSN (LPCSTR lpszFileName, LPCSTR lpszAppName, LPCSTR lpszKeyName,
    LPSTR lpszString)
{
  BOOL retcode = FALSE;
  char filename[1024];

  /* Check input parameters */
  CLEAR_ERROR ();

  /* Is a file is specified */
  if (lpszFileName)
    {
      _iodbcdm_getdsnfile (lpszFileName, filename, sizeof (filename));
      retcode =
	  WritePrivateProfileString (lpszAppName, lpszKeyName, lpszString,
	  filename);
      goto quit;
    }

  PUSH_ERROR (ODBC_ERROR_INVALID_PATH);
  goto quit;

quit:
  return retcode;
}

BOOL INSTAPI
SQLWriteFileDSNW (LPCWSTR lpszFileName, LPCWSTR lpszAppName,
    LPCWSTR lpszKeyName, LPWSTR lpszString)
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

  _string_u8 = (char *) dm_SQL_WtoU8 ((SQLWCHAR *) lpszString, SQL_NTS);
  if (_string_u8 == NULL && lpszString)
    {
      PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
      goto done;
    }

  retcode =
      SQLWriteFileDSN (_filename_u8, _appname_u8, _keyname_u8, _string_u8);

done:
  MEM_FREE (_filename_u8);
  MEM_FREE (_appname_u8);
  MEM_FREE (_keyname_u8);
  MEM_FREE (_string_u8);

  return retcode;
}
