/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 20, 2024.
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
 * File I/O stubs
 *
 * Linux and other OSs may use open64, lseek64 instead of defaulting off_t to
 * 64-bits like OSX does. This file provides cover functions to always perform
 * 64-bit file I/O.
 */

#ifndef _DKOPEN_H_
#define _DKOPEN_H_

/* Must predefine the large file flags before including sys/types.h */
#if defined (linux)
#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE
#elif defined (__APPLE__)
#else
#error Platform not recognized
#endif

#include <sys/types.h>

/* Typedef off64_t for platforms that don't have it declared */
#if defined (__APPLE__) && !defined (linux)
typedef u_int64_t off64_t;
#endif

int dkopen (const char *path, int flags, int mode);
int dkclose (int filedes);

off64_t dklseek (int fileds, off64_t offset, int whence);

#endif
