/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 14, 2025.
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
#include <sys/stat.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <warn_stat.h>

/* Utility library. */

#include <vstring.h>

/* Global library. */

#include <mail_params.h>

/* REMOVE - squirrel away a file instead of removing it */

int     REMOVE(const char *path)
{
    static VSTRING *dest;
    char   *slash;
    struct stat st;

    if (var_dont_remove == 0) {
	return (remove(path));
    } else {
	if (dest == 0)
	    dest = vstring_alloc(10);
	vstring_sprintf(dest, "saved/%s", ((slash = strrchr(path, '/')) != 0) ?
			slash + 1 : path);
	for (;;) {
	    if (stat(vstring_str(dest), &st) < 0)
		break;
	    vstring_strcat(dest, "+");
	}
	return (rename(path, vstring_str(dest)));
    }
}
