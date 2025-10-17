/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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

#include "misc.h"
#include "inifile.h"
#include "iodbc_error.h"

#ifdef _MAC
# include <getfpn.h>
#endif

#if !defined(WINDOWS) && !defined(WIN32) && !defined(OS2) && !defined(macintosh)
# include <pwd.h>
# include <unistd.h>
# include <sys/types.h>
# include <sys/stat.h>
# define UNIX_PWD
#endif

extern BOOL InstallDriverPath (LPSTR lpszPath, WORD cbPathMax,
    WORD * pcbPathOut, LPSTR envname);


BOOL
InstallDriverPathLength (WORD * pcbPathOut, LPSTR envname)
{
#ifdef _MAC
  OSErr result;
  long fldrDid;
  short fldrRef;
#endif
  BOOL retcode = FALSE;
  WORD len = 0;
  char path[1024];
  char *ptr;

#if	!defined(UNIX_PWD)
#ifdef _MAC
  result = FindFolder (kOnSystemDisk, kExtensionFolderType, kDontCreateFolder,
      &fldrRef, &fldrDid);
  if (result != noErr)
    {
      PUSH_ERROR (ODBC_ERROR_GENERAL_ERR);
      goto quit;
    }

  ptr = get_full_pathname (fldrDid, fldrRef);
  len = STRLEN (ptr);
  free (ptr);

  goto done;
#else
  /*
   *  On Windows, there is only one place to look
   */
  len = GetWindowsDirectory (path, sizeof (path));
  goto done;
#endif
#else

  /*
   *  1. Check $ODBCDRIVERS environment variable
   */
  if ((ptr = getenv (envname)) && access (ptr, R_OK | W_OK | X_OK) == 0)
    {
      len = STRLEN (ptr);
      goto done;
    }

  /*
   * 2. Check /usr/local/lib and /usr/lib
   */
#ifdef _BE
  if (access ("/boot/beos/system/lib", R_OK | W_OK | X_OK) == 0)
    {
      len = STRLEN ("/boot/beos/system/lib");
      goto done;
    }
#else
  if (access ("/usr/local/lib", R_OK | W_OK | X_OK) == 0)
    {
      len = STRLEN ("/usr/local/lib");
      goto done;
    }
#endif

#ifdef _BE
  if (access ("/boot/home/config/lib", R_OK | W_OK | X_OK) == 0)
    {
      len = STRLEN ("/boot/home/config/lib");
      goto done;
    }
#else
  if (access ("/usr/lib", R_OK | W_OK | X_OK) == 0)
    {
      len = STRLEN ("/usr/lib");
      goto done;
    }
#endif

  /*
   *  3. Check either $HOME
   */
  if (!(ptr = getenv ("HOME")))
    {
      ptr = (char *) getpwuid (getuid ());
      if (ptr)
	ptr = ((struct passwd *) ptr)->pw_dir;
    }

  if (ptr)
    {
#ifdef _BE
      sprintf (path, "%s/config/lib", ptr);
#else
      sprintf (path, "%s/lib", ptr);
#endif
      if (access (path, R_OK | W_OK | X_OK) == 0)
	{
	  len = STRLEN (path);
	  goto done;
	}
    }

  if (!mkdir (path, 0755))
    goto done;

#endif

  SQLPostInstallerError (ODBC_ERROR_GENERAL_ERR,
      "Cannot retrieve a directory where to install the driver or translator.");
  goto quit;

done:
  retcode = TRUE;

quit:
  if (pcbPathOut)
    *pcbPathOut = len;
  return retcode;
}


BOOL INSTAPI
SQLInstallDriverEx (LPCSTR lpszDriver, LPCSTR lpszPathIn, LPSTR lpszPathOut,
    WORD cbPathOutMax, WORD * pcbPathOut, WORD fRequest,
    LPDWORD lpdwUsageCount)
{
  PCONFIG pCfg = NULL, pOdbcCfg = NULL;
  BOOL retcode = FALSE;

  CLEAR_ERROR ();

  if (lpszPathIn && access (lpszPathIn, R_OK | W_OK | X_OK))
    {
      PUSH_ERROR (ODBC_ERROR_INVALID_PATH);
      goto quit;
    }

  switch (fRequest)
    {
    case ODBC_INSTALL_INQUIRY:
      if (lpszPathIn)
	{
	  if (pcbPathOut)
	    *pcbPathOut = STRLEN (lpszPathIn);
	  retcode = TRUE;
	}
      else
	retcode = InstallDriverPathLength (pcbPathOut, "ODBCDRIVERS");
      goto quit;

    case ODBC_INSTALL_COMPLETE:
      break;

    default:
      PUSH_ERROR (ODBC_ERROR_INVALID_REQUEST_TYPE);
      goto quit;
    }

  /* Check input parameters */
  if (!lpszDriver || !STRLEN (lpszDriver))
    {
      PUSH_ERROR (ODBC_ERROR_INVALID_PARAM_SEQUENCE);
      goto quit;
    }

  if (!lpszPathOut || !cbPathOutMax)
    {
      PUSH_ERROR (ODBC_ERROR_INVALID_BUFF_LEN);
      goto quit;
    }

  /* Write the out path */
  if (!InstallDriverPath (lpszPathOut, cbPathOutMax, pcbPathOut,
	  "ODBCDRIVERS"))
    goto quit;

  /* Else go through user/system odbcinst.ini */
  switch (configMode)
    {
    case ODBC_BOTH_DSN:
    case ODBC_USER_DSN:
      wSystemDSN = USERDSN_ONLY;
      break;

    case ODBC_SYSTEM_DSN:
      wSystemDSN = SYSTEMDSN_ONLY;
      break;
    }

  if (_iodbcdm_cfg_search_init (&pCfg, "odbcinst.ini", TRUE))
    {
      PUSH_ERROR (ODBC_ERROR_GENERAL_ERR);
      goto quit;
    }

  if (_iodbcdm_cfg_search_init (&pOdbcCfg, "odbc.ini", TRUE))
    {
      PUSH_ERROR (ODBC_ERROR_GENERAL_ERR);
      pOdbcCfg = NULL;
      goto done;
    }

  if (!install_from_string (pCfg, pOdbcCfg, (char *) lpszDriver, TRUE))
    {
      PUSH_ERROR (ODBC_ERROR_INVALID_KEYWORD_VALUE);
      goto done;
    }

  if (_iodbcdm_cfg_commit (pCfg) || _iodbcdm_cfg_commit (pOdbcCfg))
    {
      PUSH_ERROR (ODBC_ERROR_GENERAL_ERR);
      goto done;
    }

  retcode = TRUE;

done:
  _iodbcdm_cfg_done (pCfg);
  if (pOdbcCfg)
    _iodbcdm_cfg_done (pOdbcCfg);

quit:
  wSystemDSN = USERDSN_ONLY;
  configMode = ODBC_BOTH_DSN;

  return retcode;
}

BOOL INSTAPI
SQLInstallDriverExW (LPCWSTR lpszDriver, LPCWSTR lpszPathIn,
    LPWSTR lpszPathOut, WORD cbPathOutMax, WORD * pcbPathOut, WORD fRequest,
    LPDWORD lpdwUsageCount)
{
  char *_driver_u8 = NULL;
  char *_pathin_u8 = NULL;
  char *_pathout_u8 = NULL;
  BOOL retcode = FALSE;
  int length;
  SQLWCHAR *ptr;
  char *ptr_u8;

  for (length = 0, ptr = (SQLWCHAR *) lpszDriver; *ptr;
      length += WCSLEN (ptr) + 1, ptr += WCSLEN (ptr) + 1);

  if (length > 0)
    {
      if ((_driver_u8 = malloc (length * UTF8_MAX_CHAR_LEN + 1)) != NULL)
	{
	  for (ptr = (SQLWCHAR *) lpszDriver, ptr_u8 = _driver_u8; *ptr;
	      ptr += WCSLEN (ptr) + 1, ptr_u8 += STRLEN (ptr_u8) + 1)
	    dm_StrCopyOut2_W2A (ptr, ptr_u8, WCSLEN (ptr) * UTF8_MAX_CHAR_LEN,
		NULL);
	  *ptr_u8 = '\0';
	}
    }
  else
    _driver_u8 = (char *) dm_SQL_WtoU8 ((SQLWCHAR *) lpszDriver, SQL_NTS);

  if (_driver_u8 == NULL && lpszDriver)
    {
      PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
      goto done;
    }

  _pathin_u8 = (char *) dm_SQL_WtoU8 ((SQLWCHAR *) lpszPathIn, SQL_NTS);
  if (_pathin_u8 == NULL && lpszPathIn)
    {
      PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
      goto done;
    }

  if (cbPathOutMax > 0)
    {
      if ((_pathout_u8 =
	      malloc (cbPathOutMax * UTF8_MAX_CHAR_LEN + 1)) == NULL)
	{
	  PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
	  goto done;
	}
    }

  retcode =
      SQLInstallDriverEx (_driver_u8, _pathin_u8, _pathout_u8,
      cbPathOutMax * UTF8_MAX_CHAR_LEN, pcbPathOut, fRequest, lpdwUsageCount);

  if (retcode == TRUE)
    {
      dm_StrCopyOut2_U8toW (_pathout_u8, lpszPathOut, cbPathOutMax,
	  pcbPathOut);
    }

done:
  MEM_FREE (_driver_u8);
  MEM_FREE (_pathin_u8);
  MEM_FREE (_pathout_u8);

  return retcode;
}
