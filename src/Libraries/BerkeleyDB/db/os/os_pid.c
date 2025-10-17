/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 3, 2022.
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

#ifdef HAVE_MUTEX_SUPPORT
#include "dbinc/mutex_int.h"		/* Required to load appropriate
					   header files for thread functions. */
#endif

/*
 * __os_id --
 *	Return the current process ID.
 *
 * PUBLIC: void __os_id __P((DB_ENV *, pid_t *, db_threadid_t*));
 */
void
__os_id(dbenv, pidp, tidp)
	DB_ENV *dbenv;
	pid_t *pidp;
	db_threadid_t *tidp;
{
	/*
	 * We can't depend on dbenv not being NULL, this routine is called
	 * from places where there's no DB_ENV handle.
	 *
	 * We cache the pid in the ENV handle, getting the process ID is a
	 * fairly slow call on lots of systems.
	 */
	if (pidp != NULL) {
		if (dbenv == NULL) {
#if defined(HAVE_VXWORKS)
			*pidp = taskIdSelf();
#else
			*pidp = getpid();
#endif
		} else
			*pidp = dbenv->env->pid_cache;
	}

	if (tidp != NULL) {
#if defined(DB_WIN32)
		*tidp = GetCurrentThreadId();
#elif defined(HAVE_MUTEX_UI_THREADS)
		*tidp = thr_self();
#elif defined(HAVE_MUTEX_SOLARIS_LWP) || \
	defined(HAVE_MUTEX_PTHREADS) || defined(HAVE_PTHREAD_API)
		*tidp = pthread_self();
#else
		/*
		 * Default to just getpid.
		 */
		*tidp = 0;
#endif
	}
}
