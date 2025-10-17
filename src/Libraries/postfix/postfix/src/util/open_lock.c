/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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
#include <unistd.h>
#include <fcntl.h>

/* Utility library. */

#include <msg.h>
#include <vstream.h>
#include <vstring.h>
#include <safe_open.h>
#include <myflock.h>
#include <open_lock.h>

/* open_lock - open file and lock it for exclusive access */

VSTREAM *open_lock(const char *path, int flags, mode_t mode, VSTRING *why)
{
    VSTREAM *fp;

    /*
     * Carefully create or open the file, and lock it down. Some systems
     * don't have the O_LOCK open() flag, or the flag does not do what we
     * want, so we roll our own lock.
     */
    if ((fp = safe_open(path, flags, mode, (struct stat *) 0, -1, -1, why)) == 0)
	return (0);
    if (myflock(vstream_fileno(fp), INTERNAL_LOCK,
		MYFLOCK_OP_EXCLUSIVE | MYFLOCK_OP_NOWAIT) < 0) {
	vstring_sprintf(why, "unable to set exclusive lock: %m");
	vstream_fclose(fp);
	return (0);
    }
    return (fp);
}
