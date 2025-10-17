/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 9, 2023.
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

#if !defined(WINDOWS) && !defined(WIN32) && !defined(OS2) && !defined(macintosh)
# include <pwd.h>
# include <unistd.h>
# define UNIX_PWD
#endif

extern BOOL InstallDriverPath (LPSTR lpszPath, WORD cbPathMax,
    WORD * pcbPathOut, LPSTR envname);
extern BOOL InstallDriverPathLength (WORD * pcbPathOut, LPSTR envname);


BOOL INSTAPI
SQLInstallTranslator (LPCSTR lpszInfFile, LPCSTR lpszTranslator,
    LPCSTR lpszPathIn, LPSTR lpszPathOut, WORD cbPathOutMax,
    WORD * pcbPathOut, WORD fRequest, LPDWORD lpdwUsageCount)
{
  PCONFIG pCfg = NULL, pOdbcCfg = NULL;
  BOOL retcode = FALSE;

  /* Check input parameters */
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
	retcode = InstallDriverPathLength (pcbPathOut, "ODBCTRANSLATORS");
      goto quit;

    case ODBC_INSTALL_COMPLETE:
      break;

    default:
      PUSH_ERROR (ODBC_ERROR_INVALID_REQUEST_TYPE);
      goto quit;
    }

  /* Check input parameters */
  if (!lpszTranslator || !STRLEN (lpszTranslator))
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
	  "ODBCTRANSLATORS"))
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
	      (char *) lpszTranslator, FALSE))
	{
	  PUSH_ERROR (ODBC_ERROR_INVALID_INF);
	  goto done;
	}
    }
  else if (!install_from_string (pCfg, pOdbcCfg, (char *) lpszTranslator,
	  FALSE))
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
SQLInstallTranslatorW (LPCWSTR lpszInfFile, LPCWSTR lpszTranslator,
    LPCWSTR lpszPathIn, LPWSTR lpszPathOut, WORD cbPathOutMax,
    WORD FAR * pcbPathOut, WORD fRequest, LPDWORD lpdwUsageCount)
{
  char *_inf_u8 = NULL;
  char *_translator_u8 = NULL;
  char *_pathin_u8 = NULL;
  char *_pathout_u8 = NULL;
  BOOL retcode = FALSE;

  _inf_u8 = (char *) dm_SQL_WtoU8 ((SQLWCHAR *) lpszInfFile, SQL_NTS);
  if (_inf_u8 == NULL && lpszInfFile)
    {
      PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
      goto done;
    }

  _translator_u8 =
      (char *) dm_SQL_WtoU8 ((SQLWCHAR *) lpszTranslator, SQL_NTS);
  if (_translator_u8 == NULL && lpszTranslator)
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

  retcode = SQLInstallTranslator (_inf_u8, _translator_u8, _pathin_u8,
      _pathout_u8, cbPathOutMax * UTF8_MAX_CHAR_LEN, pcbPathOut, fRequest,
      lpdwUsageCount);

  if (retcode == TRUE)
    {
      dm_StrCopyOut2_U8toW (_pathout_u8, lpszPathOut, cbPathOutMax,
	  pcbPathOut);
    }

done:
  MEM_FREE (_inf_u8);
  MEM_FREE (_translator_u8);
  MEM_FREE (_pathin_u8);
  MEM_FREE (_pathout_u8);

  return retcode;
}
