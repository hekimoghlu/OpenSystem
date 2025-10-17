/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 12, 2022.
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
#if defined(LIBC_SCCS) && !defined(lint)
static char *rcsid = "$NetBSD: stringlist.c,v 1.2 1997/01/17 07:26:20 lukem Exp $";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/gen/stringlist.c,v 1.8 2007/01/09 00:27:55 imp Exp $");

#include "namespace.h"
#include <stdio.h>
#include <string.h>
#include <err.h>
#include <stdlib.h>
#include <stringlist.h>
#include "un-namespace.h"

#define _SL_CHUNKSIZE	20

/*
 * sl_init(): Initialize a string list
 */
StringList *
sl_init(void)
{
	StringList *sl;

	sl = malloc(sizeof(StringList));
	if (sl == NULL)
		_err(1, "stringlist: %m");

	sl->sl_cur = 0;
	sl->sl_max = _SL_CHUNKSIZE;
	sl->sl_str = malloc(sl->sl_max * sizeof(char *));
	if (sl->sl_str == NULL)
		_err(1, "stringlist: %m");
	return sl;
}


/*
 * sl_add(): Add an item to the string list
 */
int
sl_add(StringList *sl, char *name)
{
	if (sl->sl_cur == sl->sl_max - 1) {
		sl->sl_max += _SL_CHUNKSIZE;
		sl->sl_str = reallocf(sl->sl_str, sl->sl_max * sizeof(char *));
		if (sl->sl_str == NULL)
			return (-1);
	}
	sl->sl_str[sl->sl_cur++] = name;
	return (0);
}


/*
 * sl_free(): Free a stringlist
 */
void
sl_free(StringList *sl, int all)
{
	size_t i;

	if (sl == NULL)
		return;
	if (sl->sl_str) {
		if (all)
			for (i = 0; i < sl->sl_cur; i++)
				free(sl->sl_str[i]);
		free(sl->sl_str);
	}
	free(sl);
}


/*
 * sl_find(): Find a name in the string list
 */
char *
sl_find(StringList *sl, char *name)
{
	size_t i;

	for (i = 0; i < sl->sl_cur; i++)
		if (strcmp(sl->sl_str[i], name) == 0)
			return sl->sl_str[i];

	return NULL;
}
