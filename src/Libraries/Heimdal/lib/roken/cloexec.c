/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 18, 2023.
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
#include <config.h>

#include "roken.h"

void ROKEN_LIB_FUNCTION
rk_cloexec(int fd)
{
#ifdef HAVE_FCNTL
    int ret;

    ret = fcntl(fd, F_GETFD);
    if (ret == -1)
	return;
    if (fcntl(fd, F_SETFD, ret | FD_CLOEXEC) == -1)
        return;
#endif
}

void ROKEN_LIB_FUNCTION
rk_cloexec_file(FILE *f)
{
#ifdef HAVE_FCNTL
    rk_cloexec(fileno(f));
#endif
}

void ROKEN_LIB_FUNCTION
rk_cloexec_dir(DIR * d)
{
#ifndef _WIN32
    rk_cloexec(dirfd(d));
#endif
}
