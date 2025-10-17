/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 7, 2025.
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


BOOL INSTAPI
SQLRemoveTranslator (LPCSTR lpszTranslator, LPDWORD lpdwUsageCount)
{
  BOOL retcode = FALSE;
  PCONFIG pCfg;

  /* Check input parameter */
  CLEAR_ERROR ();
  if (!lpszTranslator || !STRLEN (lpszTranslator))
    {
      PUSH_ERROR (ODBC_ERROR_INVALID_NAME);
      goto quit;
    }

  if (_iodbcdm_cfg_search_init (&pCfg, "odbcinst.ini", FALSE))
    {
      PUSH_ERROR (ODBC_ERROR_REQUEST_FAILED);
      goto quit;
    }

  /* deletes the translator */
  _iodbcdm_cfg_write (pCfg, "ODBC Translators", (LPSTR) lpszTranslator, NULL);
  _iodbcdm_cfg_write (pCfg, (LPSTR) lpszTranslator, NULL, NULL);

  if (_iodbcdm_cfg_commit (pCfg))
    {
      PUSH_ERROR (ODBC_ERROR_REQUEST_FAILED);
      goto done;
    }

  retcode = TRUE;

done:
  _iodbcdm_cfg_done (pCfg);

quit:
  return retcode;
}

BOOL INSTAPI
SQLRemoveTranslatorW (LPCWSTR lpszTranslator, LPDWORD lpdwUsageCount)
{
  char *_translator_u8 = NULL;
  BOOL retcode = FALSE;

  _translator_u8 =
      (char *) dm_SQL_WtoU8 ((SQLWCHAR *) lpszTranslator, SQL_NTS);
  if (_translator_u8 == NULL && lpszTranslator)
    {
      PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
      goto done;
    }

  retcode = SQLRemoveTranslator (_translator_u8, lpdwUsageCount);

done:
  MEM_FREE (_translator_u8);

  return retcode;
}
