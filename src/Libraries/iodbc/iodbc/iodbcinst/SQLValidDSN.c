/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 10, 2023.
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

#define INVALID_CHARS	"[]{}(),;?*=!@\\"
#define INVALID_CHARSW	L"[]{}(),;?*=!@\\"

BOOL
ValidDSN (LPCSTR lpszDSN)
{
  char *currp = (char *) lpszDSN;

  while (*currp)
    {
      if (strchr (INVALID_CHARS, *currp))
	return FALSE;
      else
	currp++;
    }

  return TRUE;
}


BOOL
ValidDSNW (LPCWSTR lpszDSN)
{
  wchar_t *currp = (wchar_t *) lpszDSN;

  while (*currp)
    {
      if (wcschr (INVALID_CHARSW, *currp))
	return FALSE;
      else
	currp++;
    }

  return TRUE;
}


BOOL INSTAPI
SQLValidDSN (LPCSTR lpszDSN)
{
  BOOL retcode = FALSE;

  /* Check dsn */
  CLEAR_ERROR ();
  if (!lpszDSN || !STRLEN (lpszDSN) || STRLEN (lpszDSN) >= SQL_MAX_DSN_LENGTH)
    {
      PUSH_ERROR (ODBC_ERROR_GENERAL_ERR);
      goto quit;
    }

  retcode = ValidDSN (lpszDSN);

quit:
  return retcode;
}

BOOL INSTAPI
SQLValidDSNW (LPCWSTR lpszDSN)
{
  BOOL retcode = FALSE;

  /* Check dsn */
  CLEAR_ERROR ();
  if (!lpszDSN || !WCSLEN (lpszDSN) || WCSLEN (lpszDSN) >= SQL_MAX_DSN_LENGTH)
    {
      PUSH_ERROR (ODBC_ERROR_GENERAL_ERR);
      goto quit;
    }

  retcode = ValidDSNW (lpszDSN);

quit:
  return retcode;
}
