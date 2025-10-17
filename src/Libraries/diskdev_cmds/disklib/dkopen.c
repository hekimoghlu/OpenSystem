/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 10, 2024.
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
#include "dkopen.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>

int dkopen (const char *path, int flags, int mode)
{
#if defined (linux)
	return (open64 (path, flags, mode));
#elif defined (__APPLE__)
	return (open (path, flags, mode));
#endif
}

int dkclose (int filedes)
{
#if defined (linux)
	return (close (filedes));
#elif defined (__APPLE__)
	return (close (filedes));
#endif
}

off64_t dklseek (int filedes, off64_t offset, int whence)
{
#if defined (linux)
	return (lseek64 (filedes, offset, whence));
#elif defined (__APPLE__)
	return (lseek (filedes, offset, whence));
#endif
}

