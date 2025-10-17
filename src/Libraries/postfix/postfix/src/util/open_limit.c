/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 13, 2022.
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
/* System libraries. */

#include "sys_defs.h"
#include <sys/time.h>
#include <sys/resource.h>
#include <errno.h>

#ifdef USE_MAX_FILES_PER_PROC
#include <sys/sysctl.h>
#define MAX_FILES_PER_PROC      "kern.maxfilesperproc"
#endif

/* Application-specific. */

#include "iostuff.h"

 /*
  * 44BSD compatibility.
  */
#ifndef RLIMIT_NOFILE
#ifdef RLIMIT_OFILE
#define RLIMIT_NOFILE RLIMIT_OFILE
#endif
#endif

/* open_limit - set/query file descriptor limit */

int     open_limit(int limit)
{
#ifdef RLIMIT_NOFILE
    struct rlimit rl;
#endif

    if (limit < 0) {
	errno = EINVAL;
	return (-1);
    }
#ifdef RLIMIT_NOFILE
    if (getrlimit(RLIMIT_NOFILE, &rl) < 0)
	return (-1);
    if (limit > 0) {

	/*
	 * MacOSX incorrectly reports rlim_max as RLIM_INFINITY. The true
	 * hard limit is finite and equals the kern.maxfilesperproc value.
	 */
#ifdef USE_MAX_FILES_PER_PROC
	int     max_files_per_proc;
	size_t  len = sizeof(max_files_per_proc);

	if (sysctlbyname(MAX_FILES_PER_PROC, &max_files_per_proc, &len,
			 (void *) 0, (size_t) 0) < 0)
	    return (-1);
	if (limit > max_files_per_proc)
	    limit = max_files_per_proc;
#endif
	if (limit > rl.rlim_max)
	    rl.rlim_cur = rl.rlim_max;
	else
	    rl.rlim_cur = limit;
	if (setrlimit(RLIMIT_NOFILE, &rl) < 0)
	    return (-1);
    }
    return (rl.rlim_cur);
#endif

#ifndef RLIMIT_NOFILE
    return (getdtablesize());
#endif
}

