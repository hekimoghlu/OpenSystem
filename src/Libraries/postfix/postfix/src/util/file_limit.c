/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 12, 2022.
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
/* System library. */

#include <sys_defs.h>
#ifdef USE_ULIMIT
#include <ulimit.h>
#else
#include <sys/time.h>
#include <sys/resource.h>
#include <signal.h>
#endif
#include <limits.h>

/* Utility library. */

#include <msg.h>
#include <iostuff.h>

#define ULIMIT_BLOCK_SIZE	512

/* get_file_limit - get process-wide file size limit */

off_t   get_file_limit(void)
{
    off_t   limit;

#ifdef USE_ULIMIT
    if ((limit = ulimit(UL_GETFSIZE, 0)) < 0)
	msg_fatal("ulimit: %m");
    if (limit > OFF_T_MAX / ULIMIT_BLOCK_SIZE)
	limit = OFF_T_MAX / ULIMIT_BLOCK_SIZE;
    return (limit * ULIMIT_BLOCK_SIZE);
#else
    struct rlimit rlim;

    if (getrlimit(RLIMIT_FSIZE, &rlim) < 0)
	msg_fatal("getrlimit: %m");
    limit = rlim.rlim_cur;
    return (limit < 0 ? OFF_T_MAX : rlim.rlim_cur);
#endif						/* USE_ULIMIT */
}

/* set_file_limit - process-wide file size limit */

void    set_file_limit(off_t limit)
{
#ifdef USE_ULIMIT
    if (ulimit(UL_SETFSIZE, limit / ULIMIT_BLOCK_SIZE) < 0)
	msg_fatal("ulimit: %m");
#else
    struct rlimit rlim;

    rlim.rlim_cur = rlim.rlim_max = limit;
    if (setrlimit(RLIMIT_FSIZE, &rlim) < 0)
	msg_fatal("setrlimit: %m");
#ifdef SIGXFSZ
    if (signal(SIGXFSZ, SIG_IGN) == SIG_ERR)
	msg_fatal("signal(SIGXFSZ,SIG_IGN): %m");
#endif
#endif						/* USE_ULIMIT */
}
