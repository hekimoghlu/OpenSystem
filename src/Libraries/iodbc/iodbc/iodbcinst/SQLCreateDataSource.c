/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 29, 2022.
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
#include <iodbcadm.h>
#include <unicode.h>

#include "iodbc_error.h"
#include "dlf.h"

#if defined (__APPLE__) && !(defined (NO_FRAMEWORKS) || defined (_LP64))
#include <Carbon/Carbon.h>
#endif

extern BOOL ValidDSN (LPCSTR lpszDSN);
extern BOOL ValidDSNW (LPCWSTR lpszDSN);

#define CALL_DRVCONN_DIALBOX(path) \
	if ((handle = DLL_OPEN(path)) != NULL) \
	{ \
		if ((pDrvConn = (pDrvConnFunc)DLL_PROC(handle, "iodbcdm_drvconn_dialbox")) != NULL) \
		  pDrvConn(parent, dsn, sizeof(dsn), NULL, SQL_DRIVER_PROMPT, &config); \
      retcode = TRUE; \
		DLL_CLOSE(handle); \
	}

#define CALL_DRVCONN_DIALBOXW(path) \
	if ((handle = DLL_OPEN(path)) != NULL) \
	{ \
		if ((pDrvConnW = (pDrvConnWFunc)DLL_PROC(handle, "iodbcdm_drvconn_dialboxw")) != NULL) \
		  pDrvConnW(parent, dsn, sizeof(dsn) / sizeof(wchar_t), NULL, SQL_DRIVER_PROMPT, &config); \
      retcode = TRUE; \
		DLL_CLOSE(handle); \
	}

BOOL
CreateDataSource (HWND parent, LPCSTR lpszDSN, SQLCHAR waMode)
{
  char dsn[1024] = { 0 };
  UWORD config = ODBC_USER_DSN;
  BOOL retcode = FALSE;
  void *handle;
  pDrvConnFunc pDrvConn = NULL;
  pDrvConnWFunc pDrvConnW = NULL;
#if defined (__APPLE__) && !(defined (NO_FRAMEWORKS) || defined (_LP64))
  CFStringRef libname = NULL;
  CFBundleRef bundle;
  CFURLRef liburl;
  char name[1024] = { 0 };
#endif

  /* Load the Admin dialbox function */
#if defined (__APPLE__) && !(defined (NO_FRAMEWORKS) || defined (_LP64))
  bundle = CFBundleGetBundleWithIdentifier (CFSTR ("org.iodbc.inst"));
  if (bundle)
    {
      /* Search for the iODBCadm library */
      liburl =
	  CFBundleCopyResourceURL (bundle, CFSTR ("iODBCadm.bundle"),
	  NULL, NULL);
      if (liburl
	  && (libname =
	      CFURLCopyFileSystemPath (liburl, kCFURLPOSIXPathStyle)))
	{
	  CFStringGetCString (libname, name, sizeof (name),
	      kCFStringEncodingASCII);
	  STRCAT (name, "/Contents/MacOS/iODBCadm");
	  if (waMode == 'A')
	    {
	      CALL_DRVCONN_DIALBOX (name);
	    }
	  else
	    {
	      CALL_DRVCONN_DIALBOXW (name);
	    }
	}
      if (liburl)
	CFRelease (liburl);
      if (libname)
	CFRelease (libname);
    }
#else
  if (waMode == 'A')
    {
      CALL_DRVCONN_DIALBOX ("libiodbcadm.so");
    }
  else
    {
      CALL_DRVCONN_DIALBOXW ("libiodbcadm.so");
    }
#endif

  return retcode;
}


BOOL INSTAPI
SQLCreateDataSource_Internal (HWND hwndParent, SQLPOINTER lpszDSN,
    SQLCHAR waMode)
{
  BOOL retcode = FALSE;

  /* Check input parameters */
  CLEAR_ERROR ();
  if (!hwndParent)
    {
      PUSH_ERROR (ODBC_ERROR_INVALID_HWND);
      goto quit;
    }

  if (waMode == 'A')
    {
      if ((!lpszDSN && !ValidDSN (lpszDSN)) || (!lpszDSN
	      && !STRLEN (lpszDSN)))
	{
	  PUSH_ERROR (ODBC_ERROR_INVALID_DSN);
	  goto quit;
	}
    }
  else
    {
      if ((!lpszDSN && !ValidDSNW (lpszDSN)) || (!lpszDSN
	      && !WCSLEN (lpszDSN)))
	{
	  PUSH_ERROR (ODBC_ERROR_INVALID_DSN);
	  goto quit;
	}
    }

  retcode = CreateDataSource (hwndParent, lpszDSN, waMode);

quit:
  return retcode;
}

BOOL INSTAPI
SQLCreateDataSource (HWND hwndParent, LPCSTR lpszDSN)
{
  return SQLCreateDataSource_Internal (hwndParent, (SQLPOINTER) lpszDSN, 'A');
}

BOOL INSTAPI
SQLCreateDataSourceW (HWND hwndParent, LPCWSTR lpszDSN)
{
  return SQLCreateDataSource_Internal (hwndParent, (SQLPOINTER) lpszDSN, 'W');
}
