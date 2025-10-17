/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 31, 2025.
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
#ifndef TREE_CONFIG_H_INCLUDED
#define	TREE_CONFIG_H_INCLUDED

#if defined(PLATFORM_CONFIG_H)
/*
 * Use hand-built config.h in environments that need it.
 */
#include PLATFORM_CONFIG_H
#elif defined(HAVE_CONFIG_H)
/*
 * Most POSIX platforms use the 'configure' script to build config.h
 */
#include "../config.h"
#elif defined(__FreeBSD__)
/*
 * Built-in definitions for FreeBSD.
 */
#define	HAVE_DIRENT_D_NAMLEN 1
#define	HAVE_DIRENT_H 1
#define	HAVE_ERRNO_H 1
#define	HAVE_FCNTL_H 1
#define	HAVE_LIBARCHIVE 1
#define	HAVE_STDLIB_H 1
#define	HAVE_STRING_H 1
#define	HAVE_SYS_STAT_H 1
#define	HAVE_UNISTD_H 1
#else
/*
 * Warn if there's no platform configuration.
 */
#error Oops: No config.h and no built-in configuration in bsdtar_platform.h.
#endif /* !HAVE_CONFIG_H */

#ifdef HAVE_LIBARCHIVE
/* If we're using the platform libarchive, include system headers. */
#include <archive.h>
#include <archive_entry.h>
#else
/* Otherwise, include user headers. */
#include "archive.h"
#include "archive_entry.h"
#endif

#endif /* !TREE_CONFIG_H_INCLUDED */
