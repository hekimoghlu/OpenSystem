/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 5, 2022.
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

#include "file.h"
#ifndef lint
FILE_RCSID("@(#)$File: pread.c,v 1.3 2014/09/15 19:11:25 christos Exp $")
#endif  /* lint */
#include <fcntl.h>
#include <unistd.h>

ssize_t
pread(int fd, void *buf, size_t len, off_t off) {
	off_t old;
	ssize_t rv;

	if ((old = lseek(fd, off, SEEK_SET)) == -1)
		return -1;

	if ((rv = read(fd, buf, len)) == -1)
		return -1;

	if (lseek(fd, old, SEEK_SET) == -1)
		return -1;

	return rv;
}
