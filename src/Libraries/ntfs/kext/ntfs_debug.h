/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 27, 2024.
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
#ifndef _OSX_NTFS_DEBUG_H
#define _OSX_NTFS_DEBUG_H

#include <sys/cdefs.h>

#include "ntfs_runlist.h"

/* Forward declaration so we do not have to include <sys/mount.h> here. */
struct mount;

__private_extern__ void ntfs_debug_init(void);
__private_extern__ void ntfs_debug_deinit(void);

__private_extern__ void __ntfs_warning(const char *function,
		struct mount *mp, const char *fmt, ...) __printflike(3, 4);
#define ntfs_warning(mp, fmt, a...)	\
		__ntfs_warning(__FUNCTION__, mp, fmt, ##a)

__private_extern__ void __ntfs_error(const char *function,
		struct mount *mp, const char *fmt, ...) __printflike(3, 4);
#define ntfs_error(mp, fmt, a...)	\
		__ntfs_error(__FUNCTION__, mp, fmt, ##a)

#ifdef DEBUG

/**
 * ntfs_debug - write a debug message to the console
 * @fmt:	a printf format string containing the message
 * @...:	the variables to substitute into @fmt
 *
 * ntfs_debug() writes a message to the console but only if the driver was
 * compiled with -DDEBUG.  Otherwise, the call turns into a NOP.
 */
__private_extern__ void __ntfs_debug(const char *file, int line,
		const char *function, const char *fmt, ...)
		__printflike(4, 5);
#define ntfs_debug(fmt, a...)		\
		__ntfs_debug(__FILE__, __LINE__, __FUNCTION__, fmt, ##a)

__private_extern__ void ntfs_debug_runlist_dump(const ntfs_runlist *rl);
__private_extern__ void ntfs_debug_attr_list_dump(const u8 *al,
		const unsigned size);

#else /* !DEBUG */

#define ntfs_debug(fmt, a...)			do {} while (0)
#define ntfs_debug_runlist_dump(rl)		do {} while (0)
#define ntfs_debug_attr_list_dump(al, size)	do {} while (0)

#endif /* !DEBUG */

#endif /* !_OSX_NTFS_DEBUG_H */
