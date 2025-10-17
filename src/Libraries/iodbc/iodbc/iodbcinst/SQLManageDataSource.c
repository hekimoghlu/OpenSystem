/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 12, 2023.
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

#include "iodbc_error.h"
#include "dlf.h"

#if defined (__APPLE__) && !(defined (NO_FRAMEWORKS) || defined (_LP64))
#include <Carbon/Carbon.h>
#endif


#define CALL_ADMIN_DIALBOX(path) \
	if ((handle = DLL_OPEN(path)) != NULL) \
	{ \
		if ((pAdminBox = (pAdminBoxFunc)DLL_PROC(handle, "_iodbcdm_admin_dialbox")) != NULL) \
		  if( pAdminBox(hwndParent) == SQL_SUCCESS) \
		    retcode = TRUE; \
		DLL_CLOSE(handle); \
	} \


BOOL
ManageDataSources (HWND hwndParent)
{
  void *handle;
  pAdminBoxFunc pAdminBox;
  BOOL retcode = FALSE;
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
      if (liburl && (libname =
	      CFURLCopyFileSystemPath (liburl, kCFURLPOSIXPathStyle)))
	{
	  CFStringGetCString (libname, name, sizeof (name),
	      kCFStringEncodingASCII);
	  STRCAT (name, "/Contents/MacOS/iODBCadm");
	  CALL_ADMIN_DIALBOX (name);
	}
      if (liburl)
	CFRelease (liburl);
      if (libname)
	CFRelease (libname);
    }

#else
  CALL_ADMIN_DIALBOX ("libiodbcadm.so");
#endif

  return retcode;
}


BOOL INSTAPI
SQLManageDataSources (HWND hwndParent)
{
  BOOL retcode = FALSE;

  /* Check input parameters */
  CLEAR_ERROR ();
  if (!hwndParent)
    {
      PUSH_ERROR (ODBC_ERROR_INVALID_HWND);
      goto quit;
    }

  retcode = ManageDataSources (hwndParent);

quit:
  return retcode;
}
