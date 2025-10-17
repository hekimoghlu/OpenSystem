/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 16, 2025.
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
#include "iodbc_error.h"

#ifdef WIN32
#define SECT1			"ODBC 32 bit Data Sources"
#define SECT2			"ODBC 32 bit Drivers"
#else
#define SECT1			"ODBC Data Sources"
#define SECT2			"ODBC Drivers"
#endif

#define MAX_ENTRIES		1024

extern BOOL GetAvailableDrivers (LPCSTR lpszInfFile, LPSTR lpszBuf,
    WORD cbBufMax, WORD * pcbBufOut, BOOL infFile);

static int
SectSorter (const void *p1, const void *p2)
{
  const char **s1 = (const char **) p1;
  const char **s2 = (const char **) p2;

  return strcasecmp (*s1, *s2);
}

BOOL INSTAPI
SQLGetInstalledDrivers_Internal (LPSTR lpszBuf, WORD cbBufMax,
    WORD * pcbBufOut, SQLCHAR waMode)
{
  char buffer[4096], desc[1024], *ptr, *oldBuf = lpszBuf;
  int i, j, usernum = 0, num_entries = 0;
  void **sect = NULL;
  SQLUSMALLINT fDir = SQL_FETCH_FIRST_USER;

  if (pcbBufOut)
    *pcbBufOut = 0;

  /*
   *  Allocate the buffer for the list
   */
  if ((sect = (void **) calloc (MAX_ENTRIES, sizeof (void *))) == NULL)
    {
      PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
      return SQL_FALSE;
    }

  do
    {
      SQLSetConfigMode (fDir ==
	  SQL_FETCH_FIRST_SYSTEM ? ODBC_SYSTEM_DSN : ODBC_USER_DSN);
      SQLGetPrivateProfileString (SECT2, NULL, "", buffer,
	  sizeof (buffer) / sizeof (SQLTCHAR), "odbcinst.ini");

      /* For each drivers */
      for (ptr = buffer, i = 1; *ptr && i; ptr += STRLEN (ptr) + 1)
	{
	  /* Add this section to the datasources list */
	  if (fDir == SQL_FETCH_FIRST_SYSTEM)
	    {
	      for (j = 0; j < usernum; j++)
		{
		  if (STREQ (sect[j], ptr))
		    j = usernum;
		}
	      if (j == usernum + 1)
		continue;
	    }

	  if (num_entries >= MAX_ENTRIES)
	    {
	      i = 0;
	      break;
	    }			/* Skip the rest */

	  /* ... and its description */
	  SQLSetConfigMode (fDir ==
	      SQL_FETCH_FIRST_SYSTEM ? ODBC_SYSTEM_DSN : ODBC_USER_DSN);
	  SQLGetPrivateProfileString (SECT2, ptr, "", desc,
	      sizeof (desc) / sizeof (SQLTCHAR), "odbcinst.ini");

	  /* Check if the driver is installed */
	  if (!STRCASEEQ (desc, "Installed"))
	    continue;

	  /* Copy the driver name */
	  sect[num_entries++] = STRDUP (ptr);
	}

      switch (fDir)
	{
	case SQL_FETCH_FIRST_USER:
	  fDir = SQL_FETCH_FIRST_SYSTEM;
	  usernum = num_entries;
	  break;
	case SQL_FETCH_FIRST_SYSTEM:
	  fDir = SQL_FETCH_FIRST;
	  break;
	}
    }
  while (fDir != SQL_FETCH_FIRST);

  /*
   *  Sort all entries so we can present a nice list
   */
  if (num_entries > 1)
    {
      qsort (sect, num_entries, sizeof (char **), SectSorter);

      /* Copy back the result */
      for (i = 0; cbBufMax > 0 && i < num_entries; i++)
	{
	  if (waMode == 'A')
	    {
	      STRNCPY (lpszBuf, sect[i], cbBufMax);
	      cbBufMax -= (STRLEN (sect[i]) + 1);
	      lpszBuf += (STRLEN (sect[i]) + 1);
	    }
	  else
	    {
	      dm_StrCopyOut2_A2W (sect[i], (LPWSTR) lpszBuf, cbBufMax, NULL);
	      cbBufMax -= (STRLEN (sect[i]) + 1);
	      lpszBuf += (STRLEN (sect[i]) + 1) * sizeof (wchar_t);
	    }
	}

      if (waMode == 'A')
	*lpszBuf = '\0';
      else
	*((wchar_t *) lpszBuf) = L'\0';
    }

  /*
   *  Free old section list
   */
  if (sect)
    {
      for (i = 0; i < MAX_ENTRIES; i++)
	if (sect[i])
	  free (sect[i]);
      free (sect);
    }

  if (pcbBufOut)
    *pcbBufOut =
	lpszBuf - oldBuf + (waMode == 'A' ? sizeof (char) : sizeof (wchar_t));

  return waMode == 'A' ? (oldBuf[0] ? SQL_TRUE : SQL_FALSE) :
      (((wchar_t *) oldBuf)[0] ? SQL_TRUE : SQL_FALSE);
}

BOOL INSTAPI
SQLGetInstalledDrivers (LPSTR lpszBuf, WORD cbBufMax, WORD * pcbBufOut)
{
  return SQLGetInstalledDrivers_Internal (lpszBuf, cbBufMax, pcbBufOut, 'A');
}

BOOL INSTAPI
SQLGetInstalledDriversW (LPWSTR lpszBuf, WORD cbBufMax, WORD FAR * pcbBufOut)
{
  return SQLGetInstalledDrivers_Internal ((LPSTR) lpszBuf, cbBufMax,
      pcbBufOut, 'W');
}
