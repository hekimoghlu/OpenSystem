/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 3, 2023.
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
/* ----------- Finished and tested with shadowing ----------- */

#include <iodbc.h>
#include <odbcinst.h>
#include <iodbc_error.h>
#include <iodbcadm.h>

#include "gui.h"

BOOL INSTAPI
ConfigDSN (
    HWND	  hwndParent,
    WORD	  fRequest,
    LPCSTR	  lpszDriver,
    LPCSTR	  lpszAttributes)
{
  char *dsn = NULL, *connstr = NULL, *curr, *cour = NULL;
  char dsnread[4096] = { 0 };
  char prov[4096] = { 0 };
  int driver_type = -1, flags = 0;
  BOOL retcode = FALSE;
  UWORD confMode = ODBC_USER_DSN;

  /* Map the request User/System */
  if (fRequest < ODBC_ADD_DSN || fRequest > ODBC_REMOVE_DSN)
    {
      SQLPostInstallerError (ODBC_ERROR_INVALID_REQUEST_TYPE, NULL);
      goto done;
    }

  if (!lpszDriver || !STRLEN (lpszDriver))
    {
      SQLPostInstallerError (ODBC_ERROR_INVALID_NAME, NULL);
      goto done;
    }

  /* Retrieve the config mode */
  SQLGetConfigMode (&confMode);

  /* Retrieve the DSN if one exist */
  for (curr = (LPSTR) lpszAttributes; curr && *curr;
      curr += (STRLEN (curr) + 1))
    {
      if (!strncmp (curr, "DSN=", STRLEN ("DSN=")))
	{
	  dsn = curr + STRLEN ("DSN=");
	  break;
	}
    }

  /* Retrieve the corresponding driver */
  if (strstr (lpszDriver, "OpenLink") || strstr (lpszDriver, "Openlink")
      || strstr (lpszDriver, "oplodbc"))
    {
      driver_type = 0;

      for (curr = (LPSTR) lpszAttributes, cour = prov; curr && *curr;
	  curr += (STRLEN (curr) + 1), cour += (STRLEN (cour) + 1))
	{
	  if (!strncasecmp (curr, "Host=", STRLEN ("Host="))
	      && STRLEN (curr + STRLEN ("Host=")))
	    {
	      STRCPY (cour, curr);
	      flags |= 0x1;
	      continue;
	    }
	  if (!strncasecmp (curr, "ServerType=", STRLEN ("ServerType="))
	      && STRLEN (curr + STRLEN ("ServerType=")))
	    {
	      STRCPY (cour, curr);
	      flags |= 0x2;
	      continue;
	    }
	  STRCPY (cour, curr);
	}

      if (cour && !(flags & 1))
	{
	  STRCPY (cour, "Host=localhost\0");
	  cour += (STRLEN (cour) + 1);
	}

      if (cour && !(flags & 2))
	{
	  STRCPY (cour, "ServerType=Proxy\0");
	  cour += (STRLEN (cour) + 1);
	}

      if (cour)
	*cour = 0;
    }
  else if ((strstr (lpszDriver, "Virtuoso")
	  || strstr (lpszDriver, "virtodbc")))
    driver_type = 1;

  /* For each request */
  switch (fRequest)
    {
    case ODBC_ADD_DSN:
      /* Check if the DSN with this name already exists */

      SQLSetConfigMode (confMode);
#ifdef WIN32
      if (hwndParent && dsn
	  && SQLGetPrivateProfileString ("ODBC 32 bit Data Sources", dsn, "",
	      dsnread, sizeof (dsnread), NULL)
	  && !create_confirm (hwndParent, dsn,
	      "Are you sure you want to overwrite this DSN ?"))
#else
      if (hwndParent && dsn
	  && SQLGetPrivateProfileString ("ODBC Data Sources", dsn, "",
	      dsnread, sizeof (dsnread), NULL)
	  && !create_confirm (hwndParent, dsn,
	      "Are you sure you want to overwrite this DSN ?"))
#endif
	goto done;

      /* Call the right setup function */
      connstr =
	  create_gensetup (hwndParent, dsn,
	  STRLEN (prov) ? prov : lpszAttributes, TRUE);

      /* Check output parameters */
      if (!connstr)
	{
	  SQLPostInstallerError (ODBC_ERROR_OUT_OF_MEM, NULL);
	  goto done;
	}

      if (connstr == (LPSTR) - 1L)
	goto done;

      /* Add the DSN to the ODBC Data Sources */
      SQLSetConfigMode (confMode);
      if (!SQLWriteDSNToIni (dsn = connstr + STRLEN ("DSN="), lpszDriver))
	goto done;

      /* Add each keyword and values */
      for (curr = connstr; *curr; curr += (STRLEN (curr) + 1))
	{
	  if (strncmp (curr, "DSN=", STRLEN ("DSN=")))
	    {
	      STRCPY (dsnread, curr);
	      cour = strchr (dsnread, '=');
	      if (cour)
		*cour = 0;
	      SQLSetConfigMode (confMode);
	      if (!SQLWritePrivateProfileString (dsn, dsnread, (cour
			  && STRLEN (cour + 1)) ? cour + 1 : NULL, NULL))
		goto done;
	    }
	}

      break;

    case ODBC_CONFIG_DSN:

      if (!dsn || !STRLEN (dsn))
	{
	  SQLPostInstallerError (ODBC_ERROR_INVALID_KEYWORD_VALUE, NULL);
	  goto done;
	}

      /* Call the right setup function */
      connstr = create_gensetup (hwndParent, dsn,
	  STRLEN (prov) ? prov : lpszAttributes, FALSE);

      /* Check output parameters */
      if (!connstr)
	{
	  SQLPostInstallerError (ODBC_ERROR_OUT_OF_MEM, NULL);
	  goto done;
	}

      if (connstr == (LPSTR) - 1L)
	goto done;

      /* Compare if the DSN changed */
      if (strcmp (connstr + STRLEN ("DSN="), dsn))
	{
	  /* Remove the previous DSN */
	  SQLSetConfigMode (confMode);
	  if (!SQLRemoveDSNFromIni (dsn))
	    goto done;
	  /* Add the new DSN section */
	  SQLSetConfigMode (confMode);
	  if (!SQLWriteDSNToIni (dsn = connstr + STRLEN ("DSN="), lpszDriver))
	    goto done;
	}

      /* Add each keyword and values */
      for (curr = connstr; *curr; curr += (STRLEN (curr) + 1))
	{

	  if (strncmp (curr, "DSN=", STRLEN ("DSN=")))
	    {
	      STRCPY (dsnread, curr);
	      cour = strchr (dsnread, '=');
	      if (cour)
		*cour = 0;
	      SQLSetConfigMode (confMode);
	      if (!SQLWritePrivateProfileString (dsn, dsnread, (cour
			  && STRLEN (cour + 1)) ? cour + 1 : NULL, NULL))
		goto done;
	    }
	}

      break;

    case ODBC_REMOVE_DSN:
      if (!dsn || !STRLEN (dsn))
	{
	  SQLPostInstallerError (ODBC_ERROR_INVALID_KEYWORD_VALUE, NULL);
	  goto done;
	}

      /* Just remove the DSN */
      SQLSetConfigMode (confMode);
      if (!SQLRemoveDSNFromIni (dsn))
	goto done;
      break;
    };

quit:
  retcode = TRUE;

done:
  if (connstr && connstr != (LPSTR) - 1L && connstr != lpszAttributes
      && connstr != prov)
    free (connstr);

  return retcode;
}
