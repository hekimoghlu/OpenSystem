/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 29, 2022.
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
#pragma prototyped

#include <ast.h>

#if _lib_mkfifo

NoN(mkfifo)

#else

#include <ls.h>
#include <error.h>

#ifndef ENOSYS
#define ENOSYS	EINVAL
#endif

int
mkfifo(const char* path, mode_t mode)
{
#ifdef S_IFIFO
	return mknod(path, S_IFIFO|(mode & ~S_IFMT), 0);
#else
	errno = ENOSYS;
	return -1;
#endif
}

#endif
