/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 30, 2025.
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
#include "sysdep.h"

#include <errno.h>

#if HAVE_FCNTL_H
#include <fcntl.h>
#else
#if HAVE_SYS_FILE_H
#include <sys/file.h>
#endif
#endif

/* I basically took this from the emacs 18.57 distribution, although I
   cleaned it up a bit and made it POSIX compliant.  */

int
dup2 (oold, onew)
     int oold;
     int onew;
{
  if (oold == onew)
    return onew;
  (void) close (onew);
  
#ifdef F_DUPFD
  return fcntl (oold, F_DUPFD, onew);
#else
  {
    int onext, oret, isave;

    onext = dup (oold);
    if (onext == onew)
      return onext;
    if (onext < 0)
      return -1;
    oret = dup2 (oold, onew);
    isave = errno;
    (void) close (onext);
    errno = isave;
    return oret;
  }
#endif
}
