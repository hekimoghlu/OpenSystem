/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 5, 2024.
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
#include "uucp.h"

#include "uudefs.h"
#include "sysdep.h"
#include "system.h"

#include <errno.h>

boolean
fsysdep_make_dirs (zfile, fpublic)
     const char *zfile;
     boolean fpublic;
{
  char *zcopy, *z;
  int imode;

  zcopy = zbufcpy (zfile);

  if (fpublic)
    imode = IPUBLIC_DIRECTORY_MODE;
  else
    imode = IDIRECTORY_MODE;

  for (z = zcopy; *z != '\0'; z++)
    {
      if (*z == '/' && z != zcopy)
	{
	  /* Some versions of uuto will send a double slash.  Some
             systems will fail to create a directory ending in a
             slash.  */
	  if (z[-1] == '/')
	    continue;
	  *z = '\0';
	  if (mkdir (zcopy, imode) != 0)
	    {
	      int ierr;

	      ierr = errno;
	      if (ierr != EEXIST
		  && ierr != EISDIR
#ifdef EROFS
		  && ierr != EROFS
#endif
		  && (ierr != EACCES || ! fsysdep_directory (zcopy)))
		{
		  ulog (LOG_ERROR, "mkdir (%s): %s", zcopy,
			strerror (ierr));
		  ubuffree (zcopy);
		  return FALSE;
		}
	    }
	  *z = '/';
	}
    }

  ubuffree (zcopy);

  return TRUE;
}
