/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 2, 2023.
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
/*
 *	POSIX Standard: 4.5.2 Process Times	<sys/times.h>
 */

/*
 * If we don't have a standard system clock_t type, this must be included
 * after config.h
 */

#ifndef	_BASH_SYSTIMES_H
#define _BASH_SYSTIMES_H	1

#if defined (HAVE_SYS_TIMES_H)
#  include <sys/times.h>
#else /* !HAVE_SYS_TIMES_H */

#include <stdc.h>

/* Structure describing CPU time used by a process and its children.  */
struct tms
  {
    clock_t tms_utime;		/* User CPU time.  */
    clock_t tms_stime;		/* System CPU time.  */

    clock_t tms_cutime;		/* User CPU time of dead children.  */
    clock_t tms_cstime;		/* System CPU time of dead children.  */
  };

/* Store the CPU time used by this process and all its
   dead descendents in BUFFER.
   Return the elapsed real time from an arbitrary point in the
   past (the bash emulation uses the epoch), or (clock_t) -1 for
   errors.  All times are in CLK_TCKths of a second.  */
extern clock_t times __P((struct tms *buffer));

#endif /* !HAVE_SYS_TIMES_H */
#endif /* _BASH_SYSTIMES_H */
