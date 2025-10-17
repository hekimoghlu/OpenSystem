/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 29, 2024.
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
#include <string.h>
#include <time.h>
#include <errno.h>

#include "sudo_compat.h"
#include "sudo_debug.h"
#include "sudo_iolog.h"

/*
 * I/O log wrapper for fflush/gzflush.
 */
bool
iolog_flush(struct iolog_file *iol, const char **errstr)
{
    debug_decl(iolog_flush, SUDO_DEBUG_UTIL);
    bool ret = true;

#ifdef HAVE_ZLIB_H
    if (iol->compressed) {
	int errnum;
	if (gzflush(iol->fd.g, Z_SYNC_FLUSH) != Z_OK) {
	    if (errstr != NULL) {
		*errstr = gzerror(iol->fd.g, &errnum);
		if (errnum == Z_ERRNO)
		    *errstr = strerror(errno);
	    }
	    ret = false;
	}
    } else
#endif
    {
	if (fflush(iol->fd.f) != 0) {
	    if (errstr != NULL)
		*errstr = strerror(errno);
	    ret = false;
	}
    }

    debug_return_bool(ret);
}
