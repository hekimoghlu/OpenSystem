/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 16, 2024.
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

#include <err.h>
#include <stdlib.h>
#include <string.h>

#include "mem.h"

/*
 * malloc() wrapper.
 */
void *
sort_malloc(size_t size)
{
	void *ptr;

	if ((ptr = malloc(size)) == NULL)
		err(2, NULL);
	return (ptr);
}

/*
 * free() wrapper.
 */
void
sort_free(const void *ptr)
{

	if (ptr)
		free(__DECONST(void *, ptr));
}

/*
 * realloc() wrapper.
 */
void *
sort_realloc(void *ptr, size_t size)
{

	if ((ptr = realloc(ptr, size)) == NULL)
		err(2, NULL);
	return (ptr);
}

char *
sort_strdup(const char *str)
{
	char *dup;

	if ((dup = strdup(str)) == NULL)
		err(2, NULL);
	return (dup);
}
