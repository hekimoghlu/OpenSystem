/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 24, 2025.
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
/* $Id: stat.h,v 1.9 2009/10/01 23:48:08 tbox Exp $ */

#ifndef ISC_STAT_H
#define ISC_STAT_H 1

#include <sys/stat.h>

/*
 * Windows doesn't typedef this.
 */
typedef unsigned short mode_t;

/* open() under unix allows setting of read/write permissions
 * at the owner, group and other levels.  These don't exist in NT
 * We'll just map them all to the NT equivalent
 */

#define S_IREAD	_S_IREAD	/* read permission, owner */
#define S_IWRITE _S_IWRITE	/* write permission, owner */
#define S_IRUSR _S_IREAD	/* Owner read permission */
#define S_IWUSR _S_IWRITE	/* Owner write permission */
#define S_IRGRP _S_IREAD	/* Group read permission */
#define S_IWGRP _S_IWRITE	/* Group write permission */
#define S_IROTH _S_IREAD	/* Other read permission */
#define S_IWOTH _S_IWRITE	/* Other write permission */

#ifndef S_IFMT
# define S_IFMT   _S_IFMT
#endif
#ifndef S_IFDIR
# define S_IFDIR  _S_IFDIR
#endif
#ifndef S_IFCHR
# define S_IFCHR  _S_IFCHR
#endif
#ifndef S_IFREG
# define S_IFREG  _S_IFREG
#endif

#ifndef S_ISDIR
# define S_ISDIR(m)	(((m) & S_IFMT) == S_IFDIR)
#endif
#ifndef S_ISREG
# define S_ISREG(m)	(((m) & S_IFMT) == S_IFREG)
#endif

#endif /* ISC_STAT_H */
