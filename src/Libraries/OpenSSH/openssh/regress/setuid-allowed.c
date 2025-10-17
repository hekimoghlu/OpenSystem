/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 14, 2025.
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
/* $OpenBSD$ */

#include "includes.h"

#include <sys/types.h>
#ifdef HAVE_SYS_STATVFS_H
# include <sys/statvfs.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

static void
usage(void)
{
	fprintf(stderr, "check-setuid [path]\n");
	exit(1);
}

int
main(int argc, char **argv)
{
	const char *path = ".";
	struct statvfs sb;

	if (argc > 2)
		usage();
	else if (argc == 2)
		path = argv[1];

	if (statvfs(path, &sb) != 0) {
		/* Don't return an error if the host doesn't support statvfs */
		if (errno == ENOSYS)
			return 0;
		fprintf(stderr, "statvfs for \"%s\" failed: %s\n",
		     path, strerror(errno));
	}
	return (sb.f_flag & ST_NOSUID) ? 1 : 0;
}


