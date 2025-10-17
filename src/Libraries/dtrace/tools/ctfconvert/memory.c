/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 21, 2024.
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
/*
 * Copyright 2001-2002 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

/*
 * Routines for memory management
 */

#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

static void
memory_bailout(void)
{
	(void) fprintf(stderr, "Out of memory\n");
	exit(1);
}

void *
xmalloc(size_t size)
{
	void *mem;

	if ((mem = malloc(size)) == NULL)
		memory_bailout();

	return (mem);
}

void *
xcalloc(size_t size)
{
	void *mem;

	mem = xmalloc(size);
	bzero(mem, size);

	return (mem);
}

char *
xstrdup(const char *str)
{
	char *newstr;

	if ((newstr = strdup(str)) == NULL)
		memory_bailout();

	return (newstr);
}

char *
xstrndup(char *str, size_t len)
{
	char *newstr;

	if ((newstr = malloc(len + 1)) == NULL)
		memory_bailout();

	(void) strncpy(newstr, str, len);
	newstr[len] = '\0';

	return (newstr);
}

void *
xrealloc(void *ptr, size_t size)
{
	void *mem;

	if ((mem = realloc(ptr, size)) == NULL)
		memory_bailout();

	return (mem);
}
