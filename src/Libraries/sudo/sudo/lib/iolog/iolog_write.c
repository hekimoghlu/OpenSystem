/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 13, 2024.
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
#ifdef HAVE_STDBOOL_H
# include <stdbool.h>
#else
# include "compat/stdbool.h"
#endif
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <time.h>

#include "sudo_compat.h"
#include "sudo_debug.h"
#include "sudo_iolog.h"

/*
 * Write to an I/O log, optionally compressing.
 */
ssize_t
iolog_write(struct iolog_file *iol, const void *buf, size_t len,
    const char **errstr)
{
    ssize_t ret;
    debug_decl(iolog_write, SUDO_DEBUG_UTIL);

    if (len > UINT_MAX) {
	errno = EINVAL;
	if (errstr != NULL)
	    *errstr = strerror(errno);
	debug_return_ssize_t(-1);
    }

#ifdef HAVE_ZLIB_H
    if (iol->compressed) {
	int errnum;

	ret = gzwrite(iol->fd.g, (const voidp)buf, len);
	if (ret == 0) {
	    ret = -1;
	    if (errstr != NULL) {
		*errstr = gzerror(iol->fd.g, &errnum);
		if (errnum == Z_ERRNO)
		    *errstr = strerror(errno);
	    }
	    goto done;
	}
	if (iolog_get_flush()) {
	    if (gzflush(iol->fd.g, Z_SYNC_FLUSH) != Z_OK) {
		ret = -1;
		if (errstr != NULL) {
		    *errstr = gzerror(iol->fd.g, &errnum);
		    if (errnum == Z_ERRNO)
			*errstr = strerror(errno);
		}
		goto done;
	    }
	}
    } else
#endif
    {
	ret = fwrite(buf, 1, len, iol->fd.f);
	if (ret == 0) {
	    ret = -1;
	    if (errstr != NULL)
		*errstr = strerror(errno);
	    goto done;
	}
	if (iolog_get_flush()) {
	    if (fflush(iol->fd.f) != 0) {
		ret = -1;
		if (errstr != NULL)
		    *errstr = strerror(errno);
		goto done;
	    }
	}
    }

done:
    debug_return_ssize_t(ret);
}
