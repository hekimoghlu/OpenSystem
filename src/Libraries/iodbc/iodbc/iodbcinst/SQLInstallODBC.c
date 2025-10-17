/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 20, 2024.
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

#include "iodbc_error.h"

BOOL
InstallODBC (HWND hwndParent, LPCSTR lpszInfFile,
    LPCSTR lpszSrcPath, LPCSTR lpszDrivers)
{
  return FALSE;
}


BOOL INSTAPI
SQLInstallODBC (HWND hwndParent, LPCSTR lpszInfFile, LPCSTR lpszSrcPath,
    LPCSTR lpszDrivers)
{
  BOOL retcode = FALSE;

  /* Check input parameters */
  CLEAR_ERROR ();
  if (!lpszDrivers || !STRLEN (lpszDrivers))
    {
      PUSH_ERROR (ODBC_ERROR_INVALID_NAME);
      goto quit;
    }

  if (!lpszInfFile || !STRLEN (lpszInfFile))
    {
      PUSH_ERROR (ODBC_ERROR_INVALID_INF);
      goto quit;
    }

  retcode = InstallODBC (hwndParent, lpszInfFile, lpszSrcPath, lpszDrivers);

quit:
  return retcode;
}

BOOL INSTAPI
SQLInstallODBCW (HWND hwndParent, LPCWSTR lpszInfFile, LPCWSTR lpszSrcPath,
    LPCWSTR lpszDrivers)
{
  char *_inf_u8 = NULL;
  char *_srcpath_u8 = NULL;
  char *_drivers_u8 = NULL;
  BOOL retcode = FALSE;

  _inf_u8 = (char *) dm_SQL_WtoU8 ((SQLWCHAR *) lpszInfFile, SQL_NTS);
  if (_inf_u8 == NULL && lpszInfFile)
    {
      PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
      goto done;
    }

  _srcpath_u8 = (char *) dm_SQL_WtoU8 ((SQLWCHAR *) lpszSrcPath, SQL_NTS);
  if (_srcpath_u8 == NULL && lpszSrcPath)
    {
      PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
      goto done;
    }

  _drivers_u8 = (char *) dm_SQL_WtoU8 ((SQLWCHAR *) lpszDrivers, SQL_NTS);
  if (_drivers_u8 == NULL && lpszDrivers)
    {
      PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
      goto done;
    }

  retcode = SQLInstallODBC (hwndParent, _inf_u8, _srcpath_u8, _drivers_u8);

done:
  MEM_FREE (_inf_u8);
  MEM_FREE (_srcpath_u8);
  MEM_FREE (_drivers_u8);

  return retcode;
}
