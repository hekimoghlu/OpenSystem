/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 24, 2024.
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
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <argv.h>
#include <safe.h>
#include <clean_env.h>

/* clean_env - clean up the environment */

void    clean_env(char **preserve_list)
{
    extern char **environ;
    ARGV   *save_list;
    char   *value;
    char  **cpp;
    char   *eq;

    /*
     * Preserve or specify selected environment variables.
     */
#define STRING_AND_LENGTH(x, y) (x), (ssize_t) (y)

    save_list = argv_alloc(10);
    for (cpp = preserve_list; *cpp; cpp++)
	if ((eq = strchr(*cpp, '=')) != 0)
	    argv_addn(save_list, STRING_AND_LENGTH(*cpp, eq - *cpp),
		      STRING_AND_LENGTH(eq + 1, strlen(eq + 1)), (char *) 0);
	else if ((value = safe_getenv(*cpp)) != 0)
	    argv_add(save_list, *cpp, value, (char *) 0);

    /*
     * Truncate the process environment, if available. On some systems
     * (Ultrix!), environ can be a null pointer.
     */
    if (environ)
	environ[0] = 0;

    /*
     * Restore preserved environment variables.
     */
    for (cpp = save_list->argv; *cpp; cpp += 2)
	if (setenv(cpp[0], cpp[1], 1))
	    msg_fatal("setenv(%s, %s): %m", cpp[0], cpp[1]);

    /*
     * Cleanup.
     */
    argv_free(save_list);
}

/* update_env - apply name=value settings only */

void    update_env(char **preserve_list)
{
    char  **cpp;
    ARGV   *save_list;
    char   *eq;

    /*
     * Extract name=value settings.
     */
    save_list = argv_alloc(10);
    for (cpp = preserve_list; *cpp; cpp++)
	if ((eq = strchr(*cpp, '=')) != 0)
	    argv_addn(save_list, STRING_AND_LENGTH(*cpp, eq - *cpp),
		      STRING_AND_LENGTH(eq + 1, strlen(eq + 1)), (char *) 0);

    /*
     * Apply name=value settings.
     */
    for (cpp = save_list->argv; *cpp; cpp += 2)
	if (setenv(cpp[0], cpp[1], 1))
	    msg_fatal("setenv(%s, %s): %m", cpp[0], cpp[1]);

    /*
     * Cleanup.
     */
    argv_free(save_list);
}
