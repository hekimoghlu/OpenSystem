/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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
#include <errno.h>

/* Utility library. */

#include <msg.h>
#define WARN_STAT_INTERNAL
#include <warn_stat.h>

/* diagnose_stat - log stat warning */

static void diagnose_stat(void)
{
    struct stat st;

    /*
     * When *stat() fails with EOVERFLOW, and the interface uses 32-bit data
     * types, suggest that the program be recompiled with larger data types.
     */
#ifdef EOVERFLOW
    if (errno == EOVERFLOW && sizeof(st.st_size) == 4) {
	msg_warn("this program was built for 32-bit file handles, "
		 "but some number does not fit in 32 bits");
	msg_warn("possible solution: recompile in 64-bit mode, or "
		 "recompile in 32-bit mode with 'large file' support");
    }
#endif
}

/* warn_stat - stat with warning */

int     warn_stat(const char *path, struct stat * st)
{
    int     ret;

    ret = stat(path, st);
    if (ret < 0)
	diagnose_stat();
    return (ret);
}

/* warn_lstat - lstat with warning */

int     warn_lstat(const char *path, struct stat * st)
{
    int     ret;

    ret = lstat(path, st);
    if (ret < 0)
	diagnose_stat();
    return (ret);
}

/* warn_fstat - fstat with warning */

int     warn_fstat(int fd, struct stat * st)
{
    int     ret;

    ret = fstat(fd, st);
    if (ret < 0)
	diagnose_stat();
    return (ret);
}
