/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 25, 2022.
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

BOOL
GetAvailableDrivers (LPCSTR lpszInfFile, LPSTR lpszBuf, WORD cbBufMax,
    WORD * pcbBufOut, BOOL infFile)
{
  int sect_len = 0;
  WORD curr = 0;
  BOOL retcode = FALSE;
  PCONFIG pCfg;
  char *szId;

  if (!lpszBuf || !cbBufMax)
    {
      PUSH_ERROR (ODBC_ERROR_INVALID_BUFF_LEN);
      goto quit;
    }

  /* Open the file to read from */
  if (_iodbcdm_cfg_init (&pCfg, lpszInfFile, FALSE))
    {
      PUSH_ERROR (ODBC_ERROR_COMPONENT_NOT_FOUND);
      goto quit;
    }

  /* Get the ODBC Drivers section */
#ifdef WIN32
  if (_iodbcdm_cfg_find (pCfg, "ODBC 32 bit Drivers", NULL))
#else
  if (_iodbcdm_cfg_find (pCfg, "ODBC Drivers", NULL))
#endif
    {
      PUSH_ERROR (ODBC_ERROR_COMPONENT_NOT_FOUND);
      goto done;
    }

  while (curr < cbBufMax && 0 == _iodbcdm_cfg_nextentry (pCfg))
    {
      if (_iodbcdm_cfg_section (pCfg))
	break;

      if (_iodbcdm_cfg_define (pCfg) && pCfg->id)
	{
	  szId = pCfg->id;
	  while (infFile && *szId == '"')
	    szId++;

	  sect_len = STRLEN (szId);
	  if (!sect_len)
	    {
	      PUSH_ERROR (ODBC_ERROR_INVALID_INF);
	      goto done;
	    }

	  while (infFile && *(szId + sect_len - 1) == '"')
	    sect_len -= 1;

	  sect_len = sect_len > cbBufMax - curr ? cbBufMax - curr : sect_len;

	  if (sect_len)
	    memmove (lpszBuf + curr, szId, sect_len);
	  else
	    {
	      PUSH_ERROR (ODBC_ERROR_INVALID_INF);
	      goto done;
	    }

	  curr += sect_len;
	  lpszBuf[curr++] = 0;
	}
    }

  if (curr < cbBufMax)
    lpszBuf[curr + 1] = 0;
  if (pcbBufOut)
    *pcbBufOut = curr;
  retcode = TRUE;

done:
  _iodbcdm_cfg_done (pCfg);

quit:
  return retcode;
}


BOOL INSTAPI
SQLGetAvailableDrivers (LPCSTR lpszInfFile, LPSTR lpszBuf, WORD cbBufMax,
    WORD * pcbBufOut)
{
  BOOL retcode = FALSE;
  WORD lenBufOut;

  /* Get from the user files */
  CLEAR_ERROR ();

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

  retcode =
      GetAvailableDrivers (lpszInfFile, lpszBuf, cbBufMax, &lenBufOut, FALSE);

  if (pcbBufOut)
    *pcbBufOut = lenBufOut;

  wSystemDSN = USERDSN_ONLY;
  configMode = ODBC_BOTH_DSN;
  return retcode;
}

BOOL INSTAPI
SQLGetAvailableDriversW (LPCWSTR lpszInfFile, LPWSTR lpszBuf, WORD cbBufMax,
    WORD FAR * pcbBufOut)
{
  BOOL retcode = FALSE;
  char *_inf_u8 = NULL;
  char *_buffer_u8 = NULL;
  SQLCHAR *ptr;
  SQLWCHAR *ptrW;
  WORD len = 0, length;

  _inf_u8 = (char *) dm_SQL_WtoU8 ((SQLWCHAR *) lpszInfFile, SQL_NTS);
  if (_inf_u8 == NULL && lpszInfFile)
    {
      PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
      goto done;
    }

  if (cbBufMax > 0)
    {
      if ((_buffer_u8 = malloc (cbBufMax * UTF8_MAX_CHAR_LEN + 1)) == NULL)
	{
	  PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
	  goto done;
	}
    }

  retcode =
      SQLGetAvailableDrivers (_inf_u8, _buffer_u8,
      cbBufMax * UTF8_MAX_CHAR_LEN, pcbBufOut);

  if (retcode == TRUE)
    {
      length = 0;

      for (ptr = _buffer_u8, ptrW = lpszBuf; *ptr;
	  ptr += STRLEN (ptr) + 1, ptrW += WCSLEN (ptrW) + 1)
	{
	  dm_StrCopyOut2_U8toW (ptr, ptrW, cbBufMax - 1, &len);
	  length += len;
	}

      *ptrW = L'\0';
      if (pcbBufOut)
	*pcbBufOut = length + 1;
    }

done:
  MEM_FREE (_inf_u8);
  MEM_FREE (_buffer_u8);

  return retcode;
}
