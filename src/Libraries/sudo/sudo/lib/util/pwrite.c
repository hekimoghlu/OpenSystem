/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 7, 2024.
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

#include <errno.h>
#include <unistd.h>

#include "sudo_compat.h"

#if !defined(HAVE_PWRITE) && !defined(HAVE_PWRITE64)
ssize_t
sudo_pwrite(int fd, const void *buf, size_t nbytes, off_t offset)
{
    ssize_t nwritten;
    off_t old_offset;

    old_offset = lseek(fd, (off_t)0, SEEK_CUR);
    if (old_offset == -1 || lseek(fd, offset, SEEK_SET) == -1)
	return -1;

    nwritten = write(fd, buf, nbytes);
    if (lseek(fd, old_offset, SEEK_SET) == -1)
	return -1;

    return nwritten;
}
#endif /* HAVE_PWRITE && !HAVE_PWRITE64 */
