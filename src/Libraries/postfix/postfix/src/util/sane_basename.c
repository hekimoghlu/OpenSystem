/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 4, 2022.
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

/* Utility library. */

#include <vstring.h>
#include <stringops.h>

#define STR(x)	vstring_str(x)

/* sane_basename - skip directory prefix */

char   *sane_basename(VSTRING *bp, const char *path)
{
    static VSTRING *buf;
    const char *first;
    const char *last;

    /*
     * Your buffer or mine?
     */
    if (bp == 0) {
	bp = buf;
	if (bp == 0)
	    bp = buf = vstring_alloc(10);
    }

    /*
     * Special case: return "." for null or zero-length input.
     */
    if (path == 0 || *path == 0)
	return (STR(vstring_strcpy(bp, ".")));

    /*
     * Remove trailing '/' characters from input. Return "/" if input is all
     * '/' characters.
     */
    last = path + strlen(path) - 1;
    while (*last == '/') {
	if (last == path)
	    return (STR(vstring_strcpy(bp, "/")));
	last--;
    }

    /*
     * The pathname does not end in '/'. Skip to last '/' character if any.
     */
    first = last - 1;
    while (first >= path && *first != '/')
	first--;

    return (STR(vstring_strncpy(bp, first + 1, last - first)));
}

/* sane_dirname - keep directory prefix */

char   *sane_dirname(VSTRING *bp, const char *path)
{
    static VSTRING *buf;
    const char *last;

    /*
     * Your buffer or mine?
     */
    if (bp == 0) {
	bp = buf;
	if (bp == 0)
	    bp = buf = vstring_alloc(10);
    }

    /*
     * Special case: return "." for null or zero-length input.
     */
    if (path == 0 || *path == 0)
	return (STR(vstring_strcpy(bp, ".")));

    /*
     * Remove trailing '/' characters from input. Return "/" if input is all
     * '/' characters.
     */
    last = path + strlen(path) - 1;
    while (*last == '/') {
	if (last == path)
	    return (STR(vstring_strcpy(bp, "/")));
	last--;
    }

    /*
     * This pathname does not end in '/'. Skip to last '/' character if any.
     */
    while (last >= path && *last != '/')
	last--;
    if (last < path)				/* no '/' */
	return (STR(vstring_strcpy(bp, ".")));

    /*
     * Strip trailing '/' characters from dirname (not strictly needed).
     */
    while (last > path && *last == '/')
	last--;

    return (STR(vstring_strncpy(bp, path, last - path + 1)));
}

#ifdef TEST
#include <vstring_vstream.h>

int     main(int argc, char **argv)
{
    VSTRING *buf = vstring_alloc(10);
    char   *dir;
    char   *base;

    while (vstring_get_nonl(buf, VSTREAM_IN) > 0) {
	dir = sane_dirname((VSTRING *) 0, STR(buf));
	base = sane_basename((VSTRING *) 0, STR(buf));
	vstream_printf("input=\"%s\" dir=\"%s\" base=\"%s\"\n",
		       STR(buf), dir, base);
    }
    vstream_fflush(VSTREAM_OUT);
    vstring_free(buf);
    return (0);
}

#endif
