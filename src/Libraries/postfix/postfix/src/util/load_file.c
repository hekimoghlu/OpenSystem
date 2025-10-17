/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 29, 2022.
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
#include <time.h>

/* Utility library. */

#include <msg.h>
#include <vstream.h>
#include <iostuff.h>
#include <load_file.h>
#include <warn_stat.h>

/* load_file - load file with some prejudice */

void    load_file(const char *path, LOAD_FILE_FN action, void *context)
{
    VSTREAM *fp;
    struct stat st;
    time_t  before;
    time_t  after;

    /*
     * Read the file again if it is hot. This may result in reading a partial
     * parameter name or missing end marker when a file changes in the middle
     * of a read.
     */
    for (before = time((time_t *) 0); /* see below */ ; before = after) {
	if ((fp = vstream_fopen(path, O_RDONLY, 0)) == 0)
	    msg_fatal("open %s: %m", path);
	action(fp, context);
	if (fstat(vstream_fileno(fp), &st) < 0)
	    msg_fatal("fstat %s: %m", path);
	if (vstream_ferror(fp) || vstream_fclose(fp))
	    msg_fatal("read %s: %m", path);
	after = time((time_t *) 0);
	if (st.st_mtime < before - 1 || st.st_mtime > after)
	    break;
	if (msg_verbose)
	    msg_info("pausing to let %s cool down", path);
	doze(300000);
    }
}
