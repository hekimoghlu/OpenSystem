/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 17, 2024.
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

#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef HAVE_STDBOOL_H
# include <stdbool.h>
#else
# include "compat/stdbool.h"
#endif
#include <string.h>

#include "sudo_compat.h"
#include "sudo_debug.h"
#include "sudo_iolog.h"

/*
 * Create path and any intermediate directories.
 * If path ends in 'XXXXXX', use mkdtemp().
 */
bool
iolog_mkpath(char *path)
{
    size_t len;
    bool ret;
    debug_decl(iolog_mkpath, SUDO_DEBUG_UTIL);

    /*
     * Create path and intermediate subdirs as needed.
     * If path ends in at least 6 Xs (ala POSIX mktemp), use mkdtemp().
     * Sets iolog_gid (if it is not already set) as a side effect.
     */
    len = strlen(path);
    if (len >= 6 && strcmp(&path[len - 6], "XXXXXX") == 0)
	ret = iolog_mkdtemp(path);
    else
	ret = iolog_mkdirs(path);

    sudo_debug_printf(SUDO_DEBUG_INFO|SUDO_DEBUG_LINENO, "iolog path %s", path);

    debug_return_bool(ret);
}
