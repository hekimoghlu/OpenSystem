/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 20, 2024.
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
#include <limits.h>
#include <time.h>

#include "sudo_compat.h"
#include "sudo_debug.h"
#include "sudo_iolog.h"

/*
 * Like gets() but for struct iolog_file.
 */
char *
iolog_gets(struct iolog_file *iol, char *buf, size_t nbytes,
    const char **errstr)
{
    char *str;
    debug_decl(iolog_gets, SUDO_DEBUG_UTIL);

    if (nbytes > UINT_MAX) {
	errno = EINVAL;
	if (errstr != NULL)
	    *errstr = strerror(errno);
	debug_return_str(NULL);
    }

#ifdef HAVE_ZLIB_H
    if (iol->compressed) {
	if ((str = gzgets(iol->fd.g, buf, nbytes)) == NULL) {
	    if (errstr != NULL) {
		int errnum;
		*errstr = gzerror(iol->fd.g, &errnum);
		if (errnum == Z_ERRNO)
		    *errstr = strerror(errno);
	    }
	}
    } else
#endif
    {
	if ((str = fgets(buf, nbytes, iol->fd.f)) == NULL) {
	    if (errstr != NULL)
		*errstr = strerror(errno);
	}
    }
    debug_return_str(str);
}
