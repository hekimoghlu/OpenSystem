/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 17, 2024.
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
static char sccsid[] = "@(#)alias.c	8.3 (Berkeley) 5/4/95";
#endif
#endif /* not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: head/bin/sh/alias.c 317039 2017-04-16 22:10:02Z jilles $");

#include <stdlib.h>
#include "shell.h"
#include "output.h"
#include "error.h"
#include "memalloc.h"
#include "mystring.h"
#include "alias.h"
#include "options.h"
#include "builtins.h"

#define ATABSIZE 39

static struct alias *atab[ATABSIZE];
static int aliases;

static void setalias(const char *, const char *);
static int unalias(const char *);
static struct alias **hashalias(const char *);

static
void
setalias(const char *name, const char *val)
{
	struct alias *ap, **app;

	unalias(name);
	app = hashalias(name);
	INTOFF;
	ap = ckmalloc(sizeof (struct alias));
	ap->name = savestr(name);
	ap->val = savestr(val);
	ap->flag = 0;
	ap->next = *app;
	*app = ap;
	aliases++;
	INTON;
}

static void
freealias(struct alias *ap)
{
	ckfree(ap->name);
	ckfree(ap->val);
	ckfree(ap);
}

static int
unalias(const char *name)
{
	struct alias *ap, **app;

	app = hashalias(name);

	for (ap = *app; ap; app = &(ap->next), ap = ap->next) {
		if (equal(name, ap->name)) {
			/*
			 * if the alias is currently in use (i.e. its
			 * buffer is being used by the input routine) we
			 * just null out the name instead of freeing it.
			 * We could clear it out later, but this situation
			 * is so rare that it hardly seems worth it.
			 */
			if (ap->flag & ALIASINUSE)
				*ap->name = '\0';
			else {
				INTOFF;
				*app = ap->next;
				freealias(ap);
				INTON;
			}
			aliases--;
			return (0);
		}
	}

	return (1);
}

static void
rmaliases(void)
{
	struct alias *ap, **app;
	int i;

	INTOFF;
	for (i = 0; i < ATABSIZE; i++) {
		app = &atab[i];
		while (*app) {
			ap = *app;
			if (ap->flag & ALIASINUSE) {
				*ap->name = '\0';
				app = &(*app)->next;
			} else {
				*app = ap->next;
				freealias(ap);
			}
		}
	}
	aliases = 0;
	INTON;
}

struct alias *
lookupalias(const char *name, int check)
{
	struct alias *ap;

	if (aliases == 0)
		return (NULL);
	for (ap = *hashalias(name); ap; ap = ap->next) {
		if (equal(name, ap->name)) {
			if (check && (ap->flag & ALIASINUSE))
				return (NULL);
			return (ap);
		}
	}

	return (NULL);
}

static int
comparealiases(const void *p1, const void *p2)
{
	const struct alias *const *a1 = p1;
	const struct alias *const *a2 = p2;

	return strcmp((*a1)->name, (*a2)->name);
}

static void
printalias(const struct alias *a)
{
	out1fmt("%s=", a->name);
	out1qstr(a->val);
	out1c('\n');
}

static void
printaliases(void)
{
	int i, j;
	struct alias **sorted, *ap;

	INTOFF;
	sorted = ckmalloc(aliases * sizeof(*sorted));
	j = 0;
	for (i = 0; i < ATABSIZE; i++)
		for (ap = atab[i]; ap; ap = ap->next)
			if (*ap->name != '\0')
				sorted[j++] = ap;
	qsort(sorted, aliases, sizeof(*sorted), comparealiases);
	for (i = 0; i < aliases; i++) {
		printalias(sorted[i]);
		if (int_pending())
			break;
	}
	ckfree(sorted);
	INTON;
}

int
aliascmd(int argc __unused, char **argv __unused)
{
	char *n, *v;
	int ret = 0;
	struct alias *ap;

	nextopt("");

	if (*argptr == NULL) {
		printaliases();
		return (0);
	}
	while ((n = *argptr++) != NULL) {
		if ((v = strchr(n+1, '=')) == NULL) /* n+1: funny ksh stuff */
			if ((ap = lookupalias(n, 0)) == NULL) {
				warning("%s: not found", n);
				ret = 1;
			} else
				printalias(ap);
		else {
			*v++ = '\0';
			setalias(n, v);
		}
	}

	return (ret);
}

int
unaliascmd(int argc __unused, char **argv __unused)
{
	int i;

	while ((i = nextopt("a")) != '\0') {
		if (i == 'a') {
			rmaliases();
			return (0);
		}
	}
	for (i = 0; *argptr; argptr++)
		i |= unalias(*argptr);

	return (i);
}

static struct alias **
hashalias(const char *p)
{
	unsigned int hashval;

	hashval = (unsigned char)*p << 4;
	while (*p)
		hashval+= *p++;
	return &atab[hashval % ATABSIZE];
}
