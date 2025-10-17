/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 7, 2025.
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

extern BOOL InstallDriverPath (LPSTR lpszPath, WORD cbPathMax,
    WORD * pcbPathOut, LPSTR envname);

BOOL INSTAPI
SQLInstallDriverManager (LPSTR lpszPath, WORD cbPathMax, WORD * pcbPathOut)
{
  BOOL retcode = FALSE;

  /* Check input parameters */
  CLEAR_ERROR ();
  if (!lpszPath || !cbPathMax)
    {
      PUSH_ERROR (ODBC_ERROR_INVALID_BUFF_LEN);
      goto quit;
    }

  retcode =
      InstallDriverPath (lpszPath, cbPathMax, pcbPathOut, "ODBCMANAGER");

quit:
  return retcode;
}

BOOL INSTAPI
SQLInstallDriverManagerW (LPWSTR lpszPath, WORD cbPathMax,
    WORD FAR * pcbPathOut)
{
  char *_path_u8 = NULL;
  BOOL retcode = FALSE;

  if (cbPathMax > 0)
    {
      if ((_path_u8 = malloc (cbPathMax * UTF8_MAX_CHAR_LEN + 1)) == NULL)
	{
	  PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
	  goto done;
	}
    }

  retcode =
      SQLInstallDriverManager (_path_u8, cbPathMax * UTF8_MAX_CHAR_LEN,
      pcbPathOut);

  if (retcode == TRUE)
    {
      dm_StrCopyOut2_U8toW (_path_u8, lpszPath, cbPathMax, pcbPathOut);
    }

done:
  MEM_FREE (_path_u8);

  return retcode;
}
