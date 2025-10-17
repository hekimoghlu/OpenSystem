/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 8, 2022.
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

/*
 * __os_ctime --
 *	Format a time-stamp.
 *
 * PUBLIC: char *__os_ctime __P((const time_t *, char *));
 */
char *
__os_ctime(tod, time_buf)
	const time_t *tod;
	char *time_buf;
{
	time_buf[CTIME_BUFLEN - 1] = '\0';

	/*
	 * The ctime_r interface is the POSIX standard, thread-safe version of
	 * ctime.  However, it was implemented in three different ways (with
	 * and without a buffer length argument, and where the buffer length
	 * argument was an int vs. a size_t *).  Also, you can't depend on a
	 * return of (char *) from ctime_r, HP-UX 10.XX's version returned an
	 * int.
	 */
#if defined(HAVE_VXWORKS)
	{
	size_t buflen = CTIME_BUFLEN;
	(void)ctime_r(tod, time_buf, &buflen);
	}
#elif defined(HAVE_CTIME_R_3ARG)
	(void)ctime_r(tod, time_buf, CTIME_BUFLEN);
#elif defined(HAVE_CTIME_R)
	(void)ctime_r(tod, time_buf);
#else
	(void)strncpy(time_buf, ctime(tod), CTIME_BUFLEN - 1);
#endif
	return (time_buf);
}
