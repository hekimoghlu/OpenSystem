/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 12, 2023.
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
#ifndef lint
#if 0
static char sccsid[] = "@(#)vars.c	8.1 (Berkeley) 6/6/93";
#endif
#endif /* not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include "rcv.h"
#include "extern.h"

/*
 * Mail -- a mail program
 *
 * Variable handling stuff.
 */

/*
 * Assign a value to a variable.
 */
void
assign(const char *name, const char *value)
{
	struct var *vp;
	int h;

	h = hash(name);
	vp = lookup(name);
	if (vp == NULL) {
		if ((vp = calloc(1, sizeof(*vp))) == NULL)
			err(1, "Out of memory");
		vp->v_name = vcopy(name);
		vp->v_link = variables[h];
		variables[h] = vp;
	}
	else
		vfree(vp->v_value);
	vp->v_value = vcopy(value);
}

/*
 * Free up a variable string.  We do not bother to allocate
 * strings whose value is "" since they are expected to be frequent.
 * Thus, we cannot free same!
 */
void
vfree(char *cp)
{
	if (*cp != '\0')
		(void)free(cp);
}

/*
 * Copy a variable value into permanent (ie, not collected after each
 * command) space.  Do not bother to alloc space for ""
 */

char *
vcopy(const char *str)
{
	char *new;
	unsigned len;

	if (*str == '\0')
		return ("");
	len = strlen(str) + 1;
	if ((new = malloc(len)) == NULL)
		err(1, "Out of memory");
	bcopy(str, new, (int)len);
	return (new);
}

/*
 * Get the value of a variable and return it.
 * Look in the environment if its not available locally.
 */

char *
value(const char *name)
{
	struct var *vp;

	if ((vp = lookup(name)) == NULL)
		return (getenv(name));
	return (vp->v_value);
}

/*
 * Locate a variable and return its variable
 * node.
 */

struct var *
lookup(const char *name)
{
	struct var *vp;

	for (vp = variables[hash(name)]; vp != NULL; vp = vp->v_link)
		if (*vp->v_name == *name && equal(vp->v_name, name))
			return (vp);
	return (NULL);
}

/*
 * Locate a group name and return it.
 */

struct grouphead *
findgroup(char name[])
{
	struct grouphead *gh;

	for (gh = groups[hash(name)]; gh != NULL; gh = gh->g_link)
		if (*gh->g_name == *name && equal(gh->g_name, name))
			return (gh);
	return (NULL);
}

/*
 * Print a group out on stdout
 */
void
printgroup(char name[])
{
	struct grouphead *gh;
	struct group *gp;

	if ((gh = findgroup(name)) == NULL) {
		printf("\"%s\": not a group\n", name);
		return;
	}
	printf("%s\t", gh->g_name);
	for (gp = gh->g_list; gp != NULL; gp = gp->ge_link)
		printf(" %s", gp->ge_name);
	printf("\n");
}

/*
 * Hash the passed string and return an index into
 * the variable or group hash table.
 */
int
hash(const char *name)
{
	int h = 0;

	while (*name != '\0') {
		h <<= 2;
		h += *name++;
	}
	if (h < 0 && (h = -h) < 0)
		h = 0;
	return (h % HSHSIZE);
}
