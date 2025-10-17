/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 24, 2024.
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
#include <time.h>

#include "sudo_compat.h"
#include "sudo_debug.h"
#include "sudo_iolog.h"

/*
 * I/O log wrapper for fseek/gzseek.
 */
off_t
iolog_seek(struct iolog_file *iol, off_t offset, int whence)
{
    off_t ret;
    //debug_decl(iolog_seek, SUDO_DEBUG_UTIL);

#ifdef HAVE_ZLIB_H
    if (iol->compressed)
	ret = gzseek(iol->fd.g, offset, whence);
    else
#endif
	ret = fseeko(iol->fd.f, offset, whence);

    //debug_return_off_t(ret);
    return ret;
}

/*
 * I/O log wrapper for rewind/gzrewind.
 */
void
iolog_rewind(struct iolog_file *iol)
{
    debug_decl(iolog_rewind, SUDO_DEBUG_UTIL);

#ifdef HAVE_ZLIB_H
    if (iol->compressed)
	(void)gzrewind(iol->fd.g);
    else
#endif
	rewind(iol->fd.f);

    debug_return;
}
