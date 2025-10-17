/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 14, 2025.
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
/*-
 * SPDX-License-Identifier: BSD-2-Clause
 *
 * Copyright (c)2003 Citrus Project,
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#ifndef __APPLE__
#include "namespace.h"
#endif
#include <sys/cdefs.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifndef __APPLE__
#include "un-namespace.h"
#endif

#include "citrus_namespace.h"
#include "citrus_region.h"
#include "citrus_mmap.h"

#ifdef __APPLE__
#include <TargetConditionals.h>

#define _close close
#define _fstat fstat
#define _open open
#endif

int
_citrus_map_file(struct _citrus_region * __restrict r,
    const char * __restrict path)
{
	struct stat st;
	void *head;
	int fd, ret;
#if defined(__APPLE__) && TARGET_OS_SIMULATOR
	char rootname[PATH_MAX];
	const char *simroot;
	size_t cpylen;

	simroot = getenv("IPHONE_SIMULATOR_ROOT");

	/*
	 * Without a provided root, we'll just use the system root and the below
	 * snprintf is innocuous.
	 */
	if (simroot == NULL)
		simroot = "/";
	cpylen = snprintf(rootname, sizeof(rootname), "%s/%s", simroot, path);
	if (cpylen >= sizeof(rootname))
		return (ENAMETOOLONG);

	path = &rootname[0];
#endif
	ret = 0;

	_region_init(r, NULL, 0);

	if ((fd = _open(path, O_RDONLY | O_CLOEXEC)) == -1)
		return (errno);

	if (_fstat(fd, &st)  == -1) {
		ret = errno;
		goto error;
	}
	if (!S_ISREG(st.st_mode)) {
		ret = EOPNOTSUPP;
		goto error;
	}

	head = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_FILE|MAP_PRIVATE,
	    fd, (off_t)0);
	if (head == MAP_FAILED) {
		ret = errno;
		goto error;
	}
	_region_init(r, head, (size_t)st.st_size);

error:
	(void)_close(fd);
	return (ret);
}

void
_citrus_unmap_file(struct _citrus_region *r)
{

	if (_region_head(r) != NULL) {
		(void)munmap(_region_head(r), _region_size(r));
		_region_init(r, NULL, 0);
	}
}

