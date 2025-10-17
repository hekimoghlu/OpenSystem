/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 14, 2022.
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
fsysdep_sync (e, zmsg)
     openfile_t e;
     const char *zmsg;
{
  int o;

#if USE_STDIO
  if (fflush (e) == EOF)
    {
      ulog (LOG_ERROR, "%s: fflush: %s", zmsg, strerror (errno));
      return FALSE;
    }
#endif

#if USE_STDIO
  o = fileno (e);
#else
  o = e;
#endif

#if FSYNC_ON_CLOSE
  if (fsync (o) < 0)
    {
      ulog (LOG_ERROR, "%s: fsync: %s", zmsg, strerror (errno));
      return FALSE;
    }
#endif

  return TRUE;
}
