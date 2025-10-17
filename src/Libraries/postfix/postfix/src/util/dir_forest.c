/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 17, 2025.
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
#include <ctype.h>

/* Utility library. */

#include "msg.h"
#include "dir_forest.h"

/* dir_forest - translate base name to directory forest */

char   *dir_forest(VSTRING *buf, const char *path, int depth)
{
    const char *myname = "dir_forest";
    static VSTRING *private_buf = 0;
    int     n;
    const char *cp;
    int     ch;

    /*
     * Sanity checks.
     */
    if (*path == 0)
	msg_panic("%s: empty path", myname);
    if (depth < 1)
	msg_panic("%s: depth %d", myname, depth);

    /*
     * Your buffer or mine?
     */
    if (buf == 0) {
	if (private_buf == 0)
	    private_buf = vstring_alloc(1);
	buf = private_buf;
    }

    /*
     * Generate one or more subdirectory levels, depending on the pathname
     * contents. When the pathname is short, use underscores instead.
     * Disallow non-printable characters or characters that are special to
     * the file system.
     */
    VSTRING_RESET(buf);
    for (cp = path, n = 0; n < depth; n++) {
	if ((ch = *cp) == 0) {
	    ch = '_';
	} else {
	    if (!ISPRINT(ch) || ch == '.' || ch == '/')
		msg_panic("%s: invalid pathname: %s", myname, path);
	    cp++;
	}
	VSTRING_ADDCH(buf, ch);
	VSTRING_ADDCH(buf, '/');
    }
    VSTRING_TERMINATE(buf);

    if (msg_verbose > 1)
	msg_info("%s: %s -> %s", myname, path, vstring_str(buf));
    return (vstring_str(buf));
}
