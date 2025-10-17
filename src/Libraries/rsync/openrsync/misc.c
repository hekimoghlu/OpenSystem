/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 10, 2022.
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

#include <sys/stat.h>

#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <stdarg.h>
#if HAVE_ERR
# include <err.h>
#endif

#include "extern.h"

/* 
 * Function to assist building execv() arguments.
 */
void
addargs(arglist *args, const char *fmt, ...)
{
	va_list	 	 ap;
	char		*cp;
	unsigned int	 nalloc;
	int	 	 r;

	va_start(ap, fmt);
	r = vasprintf(&cp, fmt, ap);
	va_end(ap);
	if (r == -1)
		err(ERR_NOMEM, "addargs: argument too long");

	nalloc = args->nalloc;
	if (args->list == NULL) {
		nalloc = 32;
		args->num = 0;
	} else if (args->num+2 >= nalloc)
		nalloc *= 2;

	args->list = recallocarray(args->list, args->nalloc, nalloc,
	    sizeof(char *));
	if (!args->list)
		err(ERR_NOMEM, NULL);
	args->nalloc = nalloc;
	args->list[args->num++] = cp;
	args->list[args->num] = NULL;
}

/*
 * Only valid until the next call to addargs!
 */
const char *
getarg(arglist *args, size_t idx)
{

	if (args->list == NULL || args->num < idx)
		return NULL;
	return args->list[idx];
}

void
freeargs(arglist *args)
{
	unsigned int	 i;

	if (args->list != NULL) {
		for (i = 0; i < args->num; i++)
			free(args->list[i]);
		free(args->list);
		args->nalloc = args->num = 0;
		args->list = NULL;
	}
}

/*
 * The name is just used for diagnostic output.
 *
 * Returns 0 if the file did not pass strict mode verification, 1 if it
 * successfully passed.
 */
int
check_file_mode(const char *name, int fd)
{
	struct stat sb;

	if (fstat(fd, &sb) == -1) {
		ERR("%s: fstat", name);
		return 0;
	}

	if ((sb.st_mode & S_IRWXO) != 0) {
		ERRX("%s: strict mode violation (other permission bits set)",
		    name);
		return 0;
	}

	if (geteuid() == 0 && sb.st_uid != 0) {
		ERRX("%s: strict mode violation (root process, file not owned by root)",
		    name);
		return 0;
	}

	return 1;
}
