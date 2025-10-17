/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 13, 2022.
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
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

/* System Headers */

#include <err.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* Local headers */

#include "panic.h"
#include "privs.h"
#include "at.h"

/* External variables */

/* Global functions */

void
panic(const char *a)
{
/* Something fatal has happened, print error message and exit.
 */
	if (fcreated) {
		PRIV_START
		unlink(atfile);
		PRIV_END
	}

	errx(EXIT_FAILURE, "%s", a);
}

void
perr(const char *a)
{
/* Some operating system error; print error message and exit.
 */
	int serrno = errno;

	if (fcreated) {
		PRIV_START
		unlink(atfile);
		PRIV_END
	}

	errno = serrno;
	err(EXIT_FAILURE, "%s", a);
}

void
usage(void)
{
	/* Print usage and exit. */
    fprintf(stderr, "usage: at [-q x] [-f file] [-m] time\n"
		    "       at -c job [job ...]\n"
		    "       at [-f file] -t [[CC]YY]MMDDhhmm[.SS]\n"
		    "       at -r job [job ...]\n"
		    "       at -l -q queuename\n"
		    "       at -l [job ...]\n"
		    "       atq [-q x] [-v]\n"
		    "       atrm job [job ...]\n"
		    "       batch [-f file] [-m]\n");
    exit(EXIT_FAILURE);
}
