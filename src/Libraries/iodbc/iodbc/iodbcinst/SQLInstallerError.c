/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 27, 2022.
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

LPSTR errortable[] = {
  "",
  "General installer error",
  "Invalid buffer length",
  "Invalid window handle",
  "Invalid string parameter",
  "Invalid type of request",
  "Component not found",
  "Invalid driver or translator name",
  "Invalid keyword-value pairs",
  "Invalid DSN",
  "Invalid .INF file",
  "Request failed",
  "Invalid install path.",
  "Could not load the driver or translator setup library",
  "Invalid parameter sequence",
  "Invalid log file name.",
  "Operation canceled on user request",
  "Could not increment or decrement the component usage count",
  "Creation of the DSN failed",
  "Error during writing system information",
  "Deletion of the DSN failed",
  "Out of memory",
  "Output string truncated due to a buffer not large enough",
  "Driver- or translator-specific error",
};


RETCODE INSTAPI
SQLInstallerError (WORD iError, DWORD * pfErrorCode, LPSTR lpszErrorMsg,
    WORD cbErrorMsgMax, WORD * pcbErrorMsg)
{
  LPSTR message;
  RETCODE retcode = SQL_ERROR;

  /* Check if the index is valid to retrieve an error */
  if ((iError - 1) > numerrors)
    {
      retcode = SQL_NO_DATA;
      goto quit;
    }

  if (!lpszErrorMsg || !cbErrorMsgMax)
    goto quit;

  lpszErrorMsg[cbErrorMsgMax - 1] = 0;

  /* Copy the message error */
  message = (errormsg[iError - 1]) ?
      errormsg[iError - 1] : errortable[ierror[iError - 1]];

  if (STRLEN (message) >= cbErrorMsgMax - 1)
    {
      STRNCPY (lpszErrorMsg, message, cbErrorMsgMax - 1);
      retcode = SQL_SUCCESS_WITH_INFO;
      goto quit;
    }
  else
    STRCPY (lpszErrorMsg, message);

  if (pfErrorCode)
    *pfErrorCode = ierror[iError - 1];
  if (pcbErrorMsg)
    *pcbErrorMsg = STRLEN (lpszErrorMsg);
  retcode = SQL_SUCCESS;

quit:
  return retcode;
}

RETCODE INSTAPI
SQLInstallerErrorW (WORD iError, DWORD * pfErrorCode, LPWSTR lpszErrorMsg,
    WORD cbErrorMsgMax, WORD * pcbErrorMsg)
{
  char *_errormsg_u8 = NULL;
  RETCODE retcode = SQL_ERROR;

  if (cbErrorMsgMax > 0)
    {
      if ((_errormsg_u8 =
	      malloc (cbErrorMsgMax * UTF8_MAX_CHAR_LEN + 1)) == NULL)
	{
	  PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
	  goto done;
	}
    }

  retcode =
      SQLInstallerError (iError, pfErrorCode, _errormsg_u8,
      cbErrorMsgMax * UTF8_MAX_CHAR_LEN, pcbErrorMsg);

  if (retcode != SQL_ERROR)
    {
      dm_StrCopyOut2_U8toW (_errormsg_u8, lpszErrorMsg, cbErrorMsgMax,
	  pcbErrorMsg);
    }

done:
  MEM_FREE (_errormsg_u8);

  return retcode;
}
