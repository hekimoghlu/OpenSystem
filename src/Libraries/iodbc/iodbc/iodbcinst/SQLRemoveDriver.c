/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 15, 2024.
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
SQLRemoveDriver (LPCSTR lpszDriver, BOOL fRemoveDSN, LPDWORD lpdwUsageCount)
{
  /*char *szDriverFile = NULL; */
  BOOL retcode = FALSE;
  PCONFIG pCfg = NULL, pInstCfg = NULL;
  LPSTR entries = (LPSTR) malloc (sizeof (char) * 65535), curr;
  int len = 0, i = 0;

  /* Check input parameters */
  CLEAR_ERROR ();
  if (!lpszDriver || !STRLEN (lpszDriver))
    {
      PUSH_ERROR (ODBC_ERROR_INVALID_NAME);
      goto quit;
    }

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

  if (_iodbcdm_cfg_search_init (&pCfg, "odbc.ini", FALSE))
    {
      PUSH_ERROR (ODBC_ERROR_GENERAL_ERR);
      goto done;
    }

  if (_iodbcdm_cfg_search_init (&pInstCfg, "odbcinst.ini", FALSE))
    {
      PUSH_ERROR (ODBC_ERROR_GENERAL_ERR);
      goto done;
    }

  if (fRemoveDSN)
    {
#ifdef WIN32
      if (entries &&
	  (len = _iodbcdm_list_entries (pCfg, "ODBC 32 bit Data Sources",
		  entries, 65535)))
#else
      if (entries
	  && (len =
	      _iodbcdm_list_entries (pCfg, "ODBC Data Sources", entries,
		  65535)))
#endif
	{
	  for (curr = entries; i < len;
	      i += (STRLEN (curr) + 1), curr += (STRLEN (curr) + 1))
	    {
	      int nCursor = pCfg->cursor;

	      if (_iodbcdm_cfg_rewind (pCfg))
		{
		  PUSH_ERROR (ODBC_ERROR_GENERAL_ERR);
		  goto done;
		}

#ifdef WIN32
	      if (_iodbcdm_cfg_find (pCfg, "ODBC 32 bit Data Sources", curr))
#else
	      if (_iodbcdm_cfg_find (pCfg, "ODBC Data Sources", curr))
#endif
		{
		  if (_iodbcdm_cfg_rewind (pCfg))
		    {
		      PUSH_ERROR (ODBC_ERROR_GENERAL_ERR);
		      goto done;
		    }
		  pCfg->cursor = nCursor;
		  continue;
		}

	      if (!strcmp (pCfg->value, lpszDriver))
		{
		  if (_iodbcdm_cfg_write (pCfg, curr, NULL, NULL))
		    {
		      PUSH_ERROR (ODBC_ERROR_GENERAL_ERR);
		      goto done;
		    }

#ifdef WIN32
		  if (_iodbcdm_cfg_write (pCfg, "ODBC 32 bit Data Sources",
			  curr, NULL))
#else
		  if (_iodbcdm_cfg_write (pCfg, "ODBC Data Sources", curr,
			  NULL))
#endif
		    {
		      PUSH_ERROR (ODBC_ERROR_GENERAL_ERR);
		      goto done;
		    }
		}

	      pCfg->cursor = nCursor;
	    }
	}
    }

  if (_iodbcdm_cfg_write (pInstCfg, (char *) lpszDriver, NULL, NULL))
    {
      PUSH_ERROR (ODBC_ERROR_GENERAL_ERR);
      goto done;
    }

#ifdef WIN32
  if (_iodbcdm_cfg_write (pInstCfg, "ODBC 32 bit Drivers", lpszDriver, NULL))
#else
  if (_iodbcdm_cfg_write (pInstCfg, "ODBC Drivers", (LPSTR) lpszDriver, NULL))
#endif
    {
      PUSH_ERROR (ODBC_ERROR_GENERAL_ERR);
      goto done;
    }

  if (_iodbcdm_cfg_commit (pCfg) || _iodbcdm_cfg_commit (pInstCfg))
    {
      PUSH_ERROR (ODBC_ERROR_GENERAL_ERR);
      goto done;
    }

  retcode = TRUE;

done:
  if (pCfg)
    _iodbcdm_cfg_done (pCfg);
  if (pInstCfg)
    _iodbcdm_cfg_done (pInstCfg);
  if (entries)
    free (entries);

quit:
  wSystemDSN = USERDSN_ONLY;
  configMode = ODBC_BOTH_DSN;

  return retcode;
}

BOOL INSTAPI
SQLRemoveDriverW (LPCWSTR lpszDriver, BOOL fRemoveDSN, LPDWORD lpdwUsageCount)
{
  char *_driver_u8 = NULL;
  BOOL retcode = FALSE;

  _driver_u8 = (char *) dm_SQL_WtoU8 ((SQLWCHAR *) lpszDriver, SQL_NTS);
  if (_driver_u8 == NULL && lpszDriver)
    {
      PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
      goto done;
    }

  retcode = SQLRemoveDriver (_driver_u8, fRemoveDSN, lpdwUsageCount);

done:
  MEM_FREE (_driver_u8);

  return retcode;
}
