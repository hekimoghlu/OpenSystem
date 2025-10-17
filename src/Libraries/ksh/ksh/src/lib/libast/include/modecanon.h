/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 1, 2023.
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
/*
 * Glenn Fowler
 * AT&T Research
 *
 * canonical mode_t representation
 */

#ifndef _MODECANON_H
#define _MODECANON_H

#define X_ITYPE(m)	((m)&X_IFMT)

#define	X_IFMT		0170000
#define	X_IFWHT		0160000
#define	X_IFDOOR	0150000
#define	X_IFSOCK	0140000
#define	X_IFLNK		0120000
#define	X_IFCTG		0110000
#define	X_IFREG		0100000
#define	X_IFBLK		0060000
#define	X_IFDIR		0040000
#define	X_IFCHR		0020000
#define	X_IFIFO		0010000

#define X_IPERM		0007777
#define	X_ISUID		0004000
#define	X_ISGID		0002000
#define	X_ISVTX		0001000
#define	X_IRUSR		0000400
#define	X_IWUSR		0000200
#define	X_IXUSR		0000100
#define	X_IRGRP		0000040
#define	X_IWGRP		0000020
#define	X_IXGRP		0000010
#define	X_IROTH		0000004
#define	X_IWOTH		0000002
#define	X_IXOTH		0000001

#define X_IRWXU		(X_IRUSR|X_IWUSR|X_IXUSR)
#define X_IRWXG		(X_IRGRP|X_IWGRP|X_IXGRP)
#define X_IRWXO		(X_IROTH|X_IWOTH|X_IXOTH)

#endif
