/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 18, 2025.
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

/* Utility library. */

#include <iostuff.h>
#include <htable.h>
#include <vstream.h>
#include <attr.h>
#include <mymalloc.h>
#include <listen.h>

/* recv_pass_attr - receive connection attributes */

int     recv_pass_attr(int fd, HTABLE **attr, int timeout, ssize_t bufsize)
{
    VSTREAM *fp;
    int     stream_err;

    /*
     * Set up a temporary VSTREAM to receive the attributes.
     * 
     * XXX We use one-character reads to simplify the implementation.
     */
    fp = vstream_fdopen(fd, O_RDWR);
    vstream_control(fp,
		    CA_VSTREAM_CTL_BUFSIZE(bufsize),
		    CA_VSTREAM_CTL_TIMEOUT(timeout),
		    CA_VSTREAM_CTL_START_DEADLINE,
		    CA_VSTREAM_CTL_END);
    stream_err = (attr_scan(fp, ATTR_FLAG_NONE,
			    ATTR_TYPE_HASH, *attr = htable_create(1),
			    ATTR_TYPE_END) < 0
		  || vstream_feof(fp) || vstream_ferror(fp));
    vstream_fdclose(fp);

    /*
     * Error reporting and recovery.
     */
    if (stream_err) {
	htable_free(*attr, myfree);
	*attr = 0;
	return (-1);
    } else {
	if ((*attr)->used == 0) {
	    htable_free(*attr, myfree);
	    *attr = 0;
	}
	return (0);
    }
}
