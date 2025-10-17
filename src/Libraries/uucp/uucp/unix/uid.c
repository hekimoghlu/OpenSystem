/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 2, 2024.
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

#include <errno.h>

/* NetBSD apparently does not support setuid as required by POSIX when
   using saved setuid, so use seteuid instead.  */

#if HAVE_SETEUID
#define setuid seteuid
#endif

/* Switch to permissions of the invoking user.  */

boolean
fsuser_perms (pieuid, piegid)
     uid_t *pieuid;
     gid_t *piegid;
{
  uid_t ieuid, iuid;
  gid_t iegid, igid;

  ieuid = geteuid ();
  iuid = getuid ();
  if (pieuid != NULL)
    *pieuid = ieuid;

  iegid = getegid ();
  igid = getgid ();
  if (piegid != NULL)
    *piegid = iegid;

#if HAVE_SETREUID
  /* Swap the effective user id and the real user id.  We can then
     swap them back again when we want to return to the uucp user's
     permissions.  */
  if (setregid (iegid, igid) < 0)
    {
      ulog (LOG_ERROR, "setregid (%ld, %ld): %s",
	    (long) iegid, (long) igid, strerror (errno));
      return FALSE;
    }
  if (setreuid (ieuid, iuid) < 0)
    {
      ulog (LOG_ERROR, "setreuid (%ld, %ld): %s",
	    (long) ieuid, (long) iuid, strerror (errno));
      return FALSE;
    }
#else /* ! HAVE_SETREUID */
#if HAVE_SAVED_SETUID
  /* Set the effective user id to the real user id.  Since the
     effective user id is saved (it's the saved setuid) we will able
     to set back to it later.  If the real user id is root we will not
     be able to switch back and forth, so don't even try.  */
  if (iuid != 0)
    {
      if (setgid (igid) < 0)
	{
	  ulog (LOG_ERROR, "setgid (%ld): %s", (long) igid, strerror (errno));
	  return FALSE;
	}
      if (setuid (iuid) < 0)
	{
	  ulog (LOG_ERROR, "setuid (%ld): %s", (long) iuid, strerror (errno));
	  return FALSE;
	}
    }
#else /* ! HAVE_SAVED_SETUID */
  /* There's no way to switch between real permissions and effective
     permissions.  Just try to open the file with the uucp
     permissions.  */
#endif /* ! HAVE_SAVED_SETUID */
#endif /* ! HAVE_SETREUID */

  return TRUE;
}

/* Restore the uucp permissions.  */

/*ARGSUSED*/
boolean
fsuucp_perms (ieuid, iegid)
     long ieuid ATTRIBUTE_UNUSED;
     long iegid ATTRIBUTE_UNUSED;
{
#if HAVE_SETREUID
  /* Swap effective and real user id's back to what they were.  */
  if (! fsuser_perms ((uid_t *) NULL, (gid_t *) NULL))
    return FALSE;
#else /* ! HAVE_SETREUID */
#if HAVE_SAVED_SETUID
  /* Set ourselves back to our original effective user id.  */
  if (setgid ((gid_t) iegid) < 0)
    {
      ulog (LOG_ERROR, "setgid (%ld): %s", (long) iegid, strerror (errno));
      /* Is this error message helpful or confusing?  */
      if (errno == EPERM)
	ulog (LOG_ERROR,
	      "Probably HAVE_SAVED_SETUID in policy.h should be set to 0");
      return FALSE;
    }
  if (setuid ((uid_t) ieuid) < 0)
    {
      ulog (LOG_ERROR, "setuid (%ld): %s", (long) ieuid, strerror (errno));
      /* Is this error message helpful or confusing?  */
      if (errno == EPERM)
	ulog (LOG_ERROR,
	      "Probably HAVE_SAVED_SETUID in policy.h should be set to 0");
      return FALSE;
    }
#else /* ! HAVE_SAVED_SETUID */
  /* We didn't switch, no need to switch back.  */
#endif /* ! HAVE_SAVED_SETUID */
#endif /* ! HAVE_SETREUID */

  return TRUE;
}
