/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 5, 2023.
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
#include "db_config.h"

#include "db_int.h"

#ifdef HAVE_SYSTEM_INCLUDE_FILES
#if defined(HAVE_PSTAT_GETDYNAMIC)
#include <sys/pstat.h>
#endif
#endif

/*
 * __os_cpu_count --
 *	Return the number of CPUs.
 *
 * PUBLIC: u_int32_t __os_cpu_count __P((void));
 */
u_int32_t
__os_cpu_count()
{
#if defined(HAVE_PSTAT_GETDYNAMIC)
	/*
	 * HP/UX.
	 */
	struct pst_dynamic psd;

	return ((u_int32_t)pstat_getdynamic(&psd,
	    sizeof(psd), (size_t)1, 0) == -1 ? 1 : psd.psd_proc_cnt);
#elif defined(HAVE_SYSCONF) && defined(_SC_NPROCESSORS_ONLN)
	/*
	 * Solaris, Linux.
	 */
	long nproc;

	nproc = sysconf(_SC_NPROCESSORS_ONLN);
	return ((u_int32_t)(nproc > 1 ? nproc : 1));
#else
	return (1);
#endif
}
