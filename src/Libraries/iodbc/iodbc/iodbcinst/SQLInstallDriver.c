/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 6, 2022.
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


BOOL
InstallDriverPath (LPSTR lpszPath, WORD cbPathMax, WORD * pcbPathOut,
    LPSTR envname)
{
#ifdef _MAC
  OSErr result;
  long fldrDid;
  short fldrRef;
#endif
  BOOL retcode = FALSE;
  char *ptr;

  lpszPath[cbPathMax - 1] = 0;

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
  STRNCPY (lpszPath, ptr, cbPathMax - 1);
  free (ptr);

  if (STRLEN (ptr) >= cbPathMax)
    {
      PUSH_ERROR (ODBC_ERROR_INVALID_BUFF_LEN);
      goto quit;
    }
  else
    goto done;
#else
  /*
   *  On Windows, there is only one place to look
   */
  if (GetWindowsDirectory ((LPSTR) buf, cbPathMax) >= cbPathMax)
    {
      PUSH_ERROR (ODBC_ERROR_INVALID_BUFF_LEN);
      goto quit;
    }
  else
    goto done;
#endif
#else

  /*
   *  1. Check $ODBCDRIVERS environment variable
   */
  if ((ptr = getenv (envname)))
    if (access (ptr, R_OK | W_OK | X_OK) == 0)
      {
	STRNCPY (lpszPath, ptr, cbPathMax - 1);
	if (STRLEN (ptr) >= cbPathMax)
	  {
	    PUSH_ERROR (ODBC_ERROR_INVALID_BUFF_LEN);
	    goto quit;
	  }
	else
	  goto done;
      }

  /*
   * 2. Check /usr/local/lib and /usr/lib
   */
#ifdef _BE
  STRNCPY (lpszPath, "/boot/beos/system/lib", cbPathMax - 1);
  if (STRLEN (lpszPath) != STRLEN ("/boot/beos/system/lib"))
#else
  STRNCPY (lpszPath, "/usr/local/lib", cbPathMax - 1);
  if (STRLEN (lpszPath) != STRLEN ("/usr/local/lib"))
#endif
    {
      PUSH_ERROR (ODBC_ERROR_INVALID_BUFF_LEN);
      goto quit;
    }
  if (access (lpszPath, R_OK | W_OK | X_OK) == 0)
    goto done;

#ifdef _BE
  STRNCPY (lpszPath, "/boot/home/config/lib", cbPathMax - 1);
  if (STRLEN (lpszPath) != STRLEN ("/boot/home/config/lib"))
#else
  STRNCPY (lpszPath, "/usr/lib", cbPathMax - 1);
  if (STRLEN (lpszPath) != STRLEN ("/usr/lib"))
#endif
    {
      PUSH_ERROR (ODBC_ERROR_INVALID_BUFF_LEN);
      goto quit;
    }
  if (access (lpszPath, R_OK | W_OK | X_OK) == 0)
    goto done;

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
      sprintf (lpszPath, "%s/config/lib", ptr);
#else
      sprintf (lpszPath, "%s/lib", ptr);
#endif
      if (access (lpszPath, R_OK | W_OK | X_OK) == 0)
	goto done;
    }

  if (!mkdir (lpszPath, 0755))
    goto done;

#endif

  SQLPostInstallerError (ODBC_ERROR_GENERAL_ERR,
      "Cannot retrieve a directory where to install the driver or translator.");
  goto quit;

done:
  retcode = TRUE;

quit:
  if (pcbPathOut)
    *pcbPathOut = STRLEN (lpszPath);
  return retcode;
}


BOOL INSTAPI
SQLInstallDriver (LPCSTR lpszInfFile, LPCSTR lpszDriver, LPSTR lpszPath,
    WORD cbPathMax, WORD * pcbPathOut)
{
  PCONFIG pCfg = NULL, pOdbcCfg = NULL;
  BOOL retcode = FALSE;

  /* Check input parameters */
  CLEAR_ERROR ();
  if (!lpszDriver || !STRLEN (lpszDriver))
    {
      PUSH_ERROR (ODBC_ERROR_INVALID_PARAM_SEQUENCE);
      goto quit;
    }

  if (!lpszPath || !cbPathMax)
    {
      PUSH_ERROR (ODBC_ERROR_INVALID_BUFF_LEN);
      goto quit;
    }

  /* Write the out path */
  if (!InstallDriverPath (lpszPath, cbPathMax, pcbPathOut, "ODBCDRIVERS"))
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

  if (lpszInfFile)
    {
      if (!install_from_ini (pCfg, pOdbcCfg, (char *) lpszInfFile,
	      (char *) lpszDriver, TRUE))
	{
	  PUSH_ERROR (ODBC_ERROR_INVALID_INF);
	  goto done;
	}
    }
  else if (!install_from_string (pCfg, pOdbcCfg, (char *) lpszDriver, TRUE))
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
SQLInstallDriverW (LPCWSTR lpszInfFile, LPCWSTR lpszDriver, LPWSTR lpszPath,
    WORD cbPathMax, WORD FAR * pcbPathOut)
{
  char *_inf_u8 = NULL;
  char *_driver_u8 = NULL;
  char *_path_u8 = NULL;
  BOOL retcode = FALSE;

  _inf_u8 = (char *) dm_SQL_WtoU8 ((SQLWCHAR *) lpszInfFile, SQL_NTS);
  if (_inf_u8 == NULL && lpszInfFile)
    {
      PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
      goto done;
    }

  _driver_u8 = (char *) dm_SQL_WtoU8 ((SQLWCHAR *) lpszDriver, SQL_NTS);
  if (_driver_u8 == NULL && lpszDriver)
    {
      PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
      goto done;
    }

  if (cbPathMax > 0)
    {
      if ((_path_u8 = malloc (cbPathMax * UTF8_MAX_CHAR_LEN + 1)) == NULL)
	{
	  PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
	  goto done;
	}
    }

  retcode =
      SQLInstallDriver (_inf_u8, _driver_u8, _path_u8,
      cbPathMax * UTF8_MAX_CHAR_LEN, pcbPathOut);

  if (retcode == TRUE)
    {
      dm_StrCopyOut2_U8toW (_path_u8, lpszPath, cbPathMax, pcbPathOut);
    }

done:
  MEM_FREE (_inf_u8);
  MEM_FREE (_driver_u8);
  MEM_FREE (_path_u8);

  return retcode;
}
