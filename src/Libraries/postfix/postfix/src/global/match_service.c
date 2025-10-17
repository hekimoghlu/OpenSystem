/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
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
/* System library. */

#include <sys_defs.h>
#include <string.h>

#ifdef STRCASECMP_IN_STRINGS_H
#include <strings.h>
#endif

/* Utility library. */

#include <msg.h>
#include <argv.h>
#include <mymalloc.h>
#include <stringops.h>
#include <match_service.h>

/* match_service_compat - backwards compatibility */

static void match_service_compat(ARGV *argv)
{
    char  **cpp;
    char   *cp;

    for (cpp = argv->argv; *cpp; cpp++) {
	if (strrchr(*cpp, '/') == 0 && (cp = strrchr(*cpp, '.')) != 0)
	    *cp = '/';
    }
}

/* match_service_init - initialize pattern list */

ARGV   *match_service_init(const char *patterns)
{
    const char *delim = CHARS_COMMA_SP;
    ARGV   *list = argv_alloc(1);
    char   *saved_patterns = mystrdup(patterns);
    char   *bp = saved_patterns;
    const char *item;

    while ((item = mystrtok(&bp, delim)) != 0)
	argv_add(list, item, (char *) 0);
    argv_terminate(list);
    myfree(saved_patterns);
    match_service_compat(list);
    return (list);
}

/* match_service_init_argv - impedance adapter */

ARGV   *match_service_init_argv(char **patterns)
{
    ARGV   *list = argv_alloc(1);
    char  **cpp;

    for (cpp = patterns; *cpp; cpp++)
	argv_add(list, *cpp, (char *) 0);
    argv_terminate(list);
    match_service_compat(list);
    return (list);
}

/* match_service_match - match service name.type against pattern list */

int     match_service_match(ARGV *list, const char *name_type)
{
    const char *myname = "match_service_match";
    const char *type;
    char  **cpp;
    char   *pattern;
    int     match;

    /*
     * Quick check for empty list.
     */
    if (list->argv[0] == 0)
	return (0);

    /*
     * Sanity check.
     */
    if ((type = strrchr(name_type, '/')) == 0 || *++type == 0)
	msg_panic("%s: malformed service: \"%s\"; need \"name/type\" format",
		  myname, name_type);

    /*
     * Iterate over all patterns in the list, stop at the first match.
     */
    for (cpp = list->argv; (pattern = *cpp) != 0; cpp++) {
	if (msg_verbose)
	    msg_info("%s: %s ~? %s", myname, name_type, pattern);
	for (match = 1; *pattern == '!'; pattern++)
	    match = !match;
	if (strcasecmp(strchr(pattern, '/') ? name_type : type, pattern) == 0) {
	    if (msg_verbose)
		msg_info("%s: %s: found match", myname, name_type);
	    return (match);
	}
    }
    if (msg_verbose)
	msg_info("%s: %s: no match", myname, name_type);
    return (0);
}

/* match_service_free - release storage */

void    match_service_free(ARGV *list)
{
    argv_free(list);
}
