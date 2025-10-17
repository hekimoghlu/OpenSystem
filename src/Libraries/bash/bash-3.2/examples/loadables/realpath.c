/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 30, 2025.
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
#include "config.h"

#include <sys/types.h>
#include <sys/stat.h>

#include <stdio.h>
#ifdef HAVE_UNISTD_H
#  include <unistd.h>
#endif
#include "bashansi.h"
#include <maxpath.h>
#include <errno.h>

#include "builtins.h"
#include "shell.h"
#include "bashgetopt.h"

#ifndef errno
extern int	errno;
#endif

extern char	*sh_realpath();

realpath_builtin(list)
WORD_LIST	*list;
{
	int	opt, cflag, vflag, sflag, es;
	char	*r, realbuf[PATH_MAX], *p;
	struct stat sb;

	if (list == 0) {
		builtin_usage();
		return (EX_USAGE);
	}

	vflag = cflag = sflag = 0;
	reset_internal_getopt();
	while ((opt = internal_getopt (list, "csv")) != -1) {
		switch (opt) {
		case 'c':
			cflag = 1;
			break;
		case 's':
			sflag = 1;
			break;
		case 'v':
			vflag = 1;
			break;
		default:
			usage();
		}
	}

	list = loptend;

	if (list == 0)
		usage();

	for (es = EXECUTION_SUCCESS; list; list = list->next) {
		p = list->word->word;
		r = sh_realpath(p, realbuf);
		if (r == 0) {
			es = EXECUTION_FAILURE;
			if (sflag == 0)
				builtin_error("%s: cannot resolve: %s", p, strerror(errno));
			continue;
		}
		if (cflag && (stat(realbuf, &sb) < 0)) {
			es = EXECUTION_FAILURE;
			if (sflag == 0)
				builtin_error("%s: %s", p, strerror(errno));
			continue;
		}
		if (sflag == 0) {
			if (vflag)
				printf ("%s -> ", p);
			printf("%s\n", realbuf);
		}
	}
	return es;
}

char *realpath_doc[] = {
	"Display the canonicalized version of each PATHNAME argument, resolving",
	"symbolic links.  The -c option checks whether or not each resolved name",
	"exists.  The -s option produces no output; the exit status determines the",
	"valididty of each PATHNAME.  The -v option produces verbose output.  The",
	"exit status is 0 if each PATHNAME was resolved; non-zero otherwise.",
	(char *)NULL
};

struct builtin realpath_struct = {
	"realpath",		/* builtin name */
	realpath_builtin,	/* function implementing the builtin */
	BUILTIN_ENABLED,	/* initial flags for builtin */
	realpath_doc,		/* array of long documentation strings */
	"realpath [-csv] pathname [pathname...]",	/* usage synopsis */
	0			/* reserved for internal use */
};
