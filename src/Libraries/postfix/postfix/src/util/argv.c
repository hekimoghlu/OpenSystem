/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 19, 2023.
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
/* System libraries. */

#include <sys_defs.h>
#include <stdlib.h>			/* 44BSD stdarg.h uses abort() */
#include <stdarg.h>
#include <string.h>

/* Application-specific. */

#include "mymalloc.h"
#include "msg.h"
#include "argv.h"

/* argv_free - destroy string array */

ARGV   *argv_free(ARGV *argvp)
{
    char  **cpp;

    for (cpp = argvp->argv; cpp < argvp->argv + argvp->argc; cpp++)
	myfree(*cpp);
    myfree((void *) argvp->argv);
    myfree((void *) argvp);
    return (0);
}

/* argv_alloc - initialize string array */

ARGV   *argv_alloc(ssize_t len)
{
    ARGV   *argvp;
    ssize_t sane_len;

    /*
     * Make sure that always argvp->argc < argvp->len.
     */
    argvp = (ARGV *) mymalloc(sizeof(*argvp));
    argvp->len = 0;
    sane_len = (len < 2 ? 2 : len);
    argvp->argv = (char **) mymalloc((sane_len + 1) * sizeof(char *));
    argvp->len = sane_len;
    argvp->argc = 0;
    argvp->argv[0] = 0;
    return (argvp);
}

static int argv_cmp(const void *e1, const void *e2)
{
    const char *s1 = *(const char **) e1;
    const char *s2 = *(const char **) e2;

    return strcmp(s1, s2);
}

/* argv_sort - sort array in place */

ARGV   *argv_sort(ARGV *argvp)
{
    qsort(argvp->argv, argvp->argc, sizeof(argvp->argv[0]), argv_cmp);
    return (argvp);
}

/* argv_extend - extend array */

static void argv_extend(ARGV *argvp)
{
    ssize_t new_len;

    new_len = argvp->len * 2;
    argvp->argv = (char **)
	myrealloc((void *) argvp->argv, (new_len + 1) * sizeof(char *));
    argvp->len = new_len;
}

/* argv_add - add string to vector */

void    argv_add(ARGV *argvp,...)
{
    char   *arg;
    va_list ap;

    /*
     * Make sure that always argvp->argc < argvp->len.
     */
#define ARGV_SPACE_LEFT(a) ((a)->len - (a)->argc - 1)

    va_start(ap, argvp);
    while ((arg = va_arg(ap, char *)) != 0) {
	if (ARGV_SPACE_LEFT(argvp) <= 0)
	    argv_extend(argvp);
	argvp->argv[argvp->argc++] = mystrdup(arg);
    }
    va_end(ap);
    argvp->argv[argvp->argc] = 0;
}

/* argv_addn - add string to vector */

void    argv_addn(ARGV *argvp,...)
{
    char   *arg;
    ssize_t len;
    va_list ap;

    /*
     * Make sure that always argvp->argc < argvp->len.
     */
    va_start(ap, argvp);
    while ((arg = va_arg(ap, char *)) != 0) {
	if ((len = va_arg(ap, ssize_t)) < 0)
	    msg_panic("argv_addn: bad string length %ld", (long) len);
	if (ARGV_SPACE_LEFT(argvp) <= 0)
	    argv_extend(argvp);
	argvp->argv[argvp->argc++] = mystrndup(arg, len);
    }
    va_end(ap);
    argvp->argv[argvp->argc] = 0;
}

/* argv_terminate - terminate string array */

void    argv_terminate(ARGV *argvp)
{

    /*
     * Trust that argvp->argc < argvp->len.
     */
    argvp->argv[argvp->argc] = 0;
}

/* argv_truncate - truncate string array */

void    argv_truncate(ARGV *argvp, ssize_t len)
{
    char  **cpp;

    /*
     * Sanity check.
     */
    if (len < 0)
	msg_panic("argv_truncate: bad length %ld", (long) len);

    if (len < argvp->argc) {
	for (cpp = argvp->argv + len; cpp < argvp->argv + argvp->argc; cpp++)
	    myfree(*cpp);
	argvp->argc = len;
	argvp->argv[argvp->argc] = 0;
    }
}

/* argv_insert_one - insert one string into array */

void    argv_insert_one(ARGV *argvp, ssize_t where, const char *arg)
{
    ssize_t pos;

    /*
     * Sanity check.
     */
    if (where < 0 || where > argvp->argc)
	msg_panic("argv_insert_one bad position: %ld", (long) where);

    if (ARGV_SPACE_LEFT(argvp) <= 0)
	argv_extend(argvp);
    for (pos = argvp->argc; pos >= where; pos--)
	argvp->argv[pos + 1] = argvp->argv[pos];
    argvp->argv[where] = mystrdup(arg);
    argvp->argc += 1;
}

/* argv_replace_one - replace one string in array */

void    argv_replace_one(ARGV *argvp, ssize_t where, const char *arg)
{
    char   *temp;

    /*
     * Sanity check.
     */
    if (where < 0 || where >= argvp->argc)
	msg_panic("argv_replace_one bad position: %ld", (long) where);

    temp = argvp->argv[where];
    argvp->argv[where] = mystrdup(arg);
    myfree(temp);
}

/* argv_delete - remove string(s) from array */

void    argv_delete(ARGV *argvp, ssize_t first, ssize_t how_many)
{
    ssize_t pos;

    /*
     * Sanity check.
     */
    if (first < 0 || how_many < 0 || first + how_many > argvp->argc)
	msg_panic("argv_delete bad range: (start=%ld count=%ld)",
		  (long) first, (long) how_many);

    for (pos = first; pos < first + how_many; pos++)
	myfree(argvp->argv[pos]);
    for (pos = first; pos <= argvp->argc - how_many; pos++)
	argvp->argv[pos] = argvp->argv[pos + how_many];
    argvp->argc -= how_many;
}
