/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 8, 2022.
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

#if HAVE_TIME_H
#include <time.h>
#endif

#include "system.h"

#ifndef time
extern time_t time ();
#endif

/* Get the time in seconds since the epoch, with optional
   microseconds.  We use ixsysdep_process_time to get the microseconds
   if it will work (it won't if it uses times, since that returns a
   time based only on the process).  */

long
ixsysdep_time (pimicros)
     long *pimicros;
{
#if HAVE_GETTIMEOFDAY || HAVE_FTIME
  return ixsysdep_process_time (pimicros);
#else
  if (pimicros != NULL)
    *pimicros = 0;
  return (long) time ((time_t *) NULL);
#endif
}
