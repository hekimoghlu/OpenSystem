/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 22, 2023.
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
/*
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#ifdef HAVE_STDBOOL_H
# include <stdbool.h>
#else
# include "compat/stdbool.h"
#endif
#include <string.h>
#include <errno.h>
#include <time.h>

#include "sudo_compat.h"
#include "sudo_debug.h"
#include "sudo_iolog.h"

/*
 * Close an I/O log.
 */
bool
iolog_close(struct iolog_file *iol, const char **errstr)
{
    bool ret = true;
    debug_decl(iolog_close, SUDO_DEBUG_UTIL);

#ifdef HAVE_ZLIB_H
    if (iol->compressed) {
	int errnum;

	/* Must check error indicator before closing. */
	if (iol->writable) {
	    if (gzflush(iol->fd.g, Z_SYNC_FLUSH) != Z_OK) {
		ret = false;
		if (errstr != NULL) {
		    *errstr = gzerror(iol->fd.g, &errnum);
		    if (errnum == Z_ERRNO)
			*errstr = strerror(errno);
		}
	    }
	}
	errnum = gzclose(iol->fd.g);
	if (ret && errnum != Z_OK) {
	    ret = false;
	    if (errstr != NULL)
		*errstr = errnum == Z_ERRNO ? strerror(errno) : "unknown error";
	}
    } else
#endif
    if (fclose(iol->fd.f) != 0) {
	ret = false;
	if (errstr != NULL)
	    *errstr = strerror(errno);
    }

    debug_return_bool(ret);
}
