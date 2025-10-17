/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 25, 2023.
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

RETCODE INSTAPI
SQLPostInstallerError (DWORD fErrorCode, LPSTR szErrorMsg)
{
  RETCODE retcode = SQL_ERROR;

  /* Check if the index is valid to retrieve an error */
  if (fErrorCode < ODBC_ERROR_GENERAL_ERR
      || fErrorCode > ODBC_ERROR_DRIVER_SPECIFIC)
    goto quit;

  if (numerrors < ERROR_NUM)
    {
      ierror[++numerrors] = fErrorCode;
      errormsg[numerrors] = szErrorMsg;;
    }

  retcode = SQL_SUCCESS;

quit:
  return retcode;
}

RETCODE INSTAPI
SQLPostInstallerErrorW (DWORD fErrorCode, LPWSTR szErrorMsg)
{
  char *_errormsg_u8 = NULL;
  RETCODE retcode = SQL_ERROR;

  _errormsg_u8 = (char *) dm_SQL_WtoU8 ((SQLWCHAR *) szErrorMsg, SQL_NTS);
  if (_errormsg_u8 == NULL && szErrorMsg)
    {
      PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
      goto done;
    }

  retcode = SQLPostInstallerError (fErrorCode, _errormsg_u8);

done:
  MEM_FREE (_errormsg_u8);

  return retcode;
}
