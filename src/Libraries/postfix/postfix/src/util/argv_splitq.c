/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 15, 2024.
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
#include <string.h>

/* Application-specific. */

#include "mymalloc.h"
#include "stringops.h"
#include "argv.h"
#include "msg.h"

/* argv_splitq - split string into token array */

ARGV   *argv_splitq(const char *string, const char *delim, const char *parens)
{
    ARGV   *argvp = argv_alloc(1);
    char   *saved_string = mystrdup(string);
    char   *bp = saved_string;
    char   *arg;

    while ((arg = mystrtokq(&bp, delim, parens)) != 0)
	argv_add(argvp, arg, (char *) 0);
    argv_terminate(argvp);
    myfree(saved_string);
    return (argvp);
}

/* argv_splitq_count - split string into token array */

ARGV   *argv_splitq_count(const char *string, const char *delim,
			          const char *parens, ssize_t count)
{
    ARGV   *argvp = argv_alloc(1);
    char   *saved_string = mystrdup(string);
    char   *bp = saved_string;
    char   *arg;

    if (count < 1)
	msg_panic("argv_splitq_count: bad count: %ld", (long) count);
    while (count-- > 1 && (arg = mystrtokq(&bp, delim, parens)) != 0)
	argv_add(argvp, arg, (char *) 0);
    if (*bp)
	bp += strspn(bp, delim);
    if (*bp)
	argv_add(argvp, bp, (char *) 0);
    argv_terminate(argvp);
    myfree(saved_string);
    return (argvp);
}

/* argv_splitq_append - split string into token array, append to array */

ARGV   *argv_splitq_append(ARGV *argvp, const char *string, const char *delim,
			           const char *parens)
{
    char   *saved_string = mystrdup(string);
    char   *bp = saved_string;
    char   *arg;

    while ((arg = mystrtokq(&bp, delim, parens)) != 0)
	argv_add(argvp, arg, (char *) 0);
    argv_terminate(argvp);
    myfree(saved_string);
    return (argvp);
}
