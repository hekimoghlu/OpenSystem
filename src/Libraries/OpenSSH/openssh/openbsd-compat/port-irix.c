/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 22, 2023.
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
#include "includes.h"

#if defined(WITH_IRIX_PROJECT) || \
    defined(WITH_IRIX_JOBS) || \
    defined(WITH_IRIX_ARRAY)

#include <errno.h>
#include <string.h>
#include <unistd.h>

#ifdef WITH_IRIX_PROJECT
# include <proj.h>
#endif /* WITH_IRIX_PROJECT */
#ifdef WITH_IRIX_JOBS
# include <sys/resource.h>
#endif
#ifdef WITH_IRIX_AUDIT
# include <sat.h>
#endif /* WITH_IRIX_AUDIT */

#include "log.h"

void
irix_setusercontext(struct passwd *pw)
{
#ifdef WITH_IRIX_PROJECT
	prid_t projid;
#endif
#ifdef WITH_IRIX_JOBS
	jid_t jid = 0;
#elif defined(WITH_IRIX_ARRAY)
	int jid = 0;
#endif

#ifdef WITH_IRIX_JOBS
	jid = jlimit_startjob(pw->pw_name, pw->pw_uid, "interactive");
	if (jid == -1)
		fatal("Failed to create job container: %.100s",
		    strerror(errno));
#endif /* WITH_IRIX_JOBS */
#ifdef WITH_IRIX_ARRAY
	/* initialize array session */
	if (jid == 0  && newarraysess() != 0)
		fatal("Failed to set up new array session: %.100s",
		    strerror(errno));
#endif /* WITH_IRIX_ARRAY */
#ifdef WITH_IRIX_PROJECT
	/* initialize irix project info */
	if ((projid = getdfltprojuser(pw->pw_name)) == -1) {
		debug("Failed to get project id, using projid 0");
		projid = 0;
	}
	if (setprid(projid))
		fatal("Failed to initialize project %d for %s: %.100s",
		    (int)projid, pw->pw_name, strerror(errno));
#endif /* WITH_IRIX_PROJECT */
#ifdef WITH_IRIX_AUDIT
	if (sysconf(_SC_AUDIT)) {
		debug("Setting sat id to %d", (int) pw->pw_uid);
		if (satsetid(pw->pw_uid))
			debug("error setting satid: %.100s", strerror(errno));
	}
#endif /* WITH_IRIX_AUDIT */
}


#endif /* defined(WITH_IRIX_PROJECT) || defined(WITH_IRIX_JOBS) || defined(WITH_IRIX_ARRAY) */
