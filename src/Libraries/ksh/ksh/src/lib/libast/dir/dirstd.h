/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 8, 2024.
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
 * AT&T Bell Laboratories
 *
 * <dirent.h> for systems with no opendir()
 */

#ifndef _DIRENT_H
#define _DIRENT_H

typedef struct
{
	int		dd_fd;		/* file descriptor		*/
#ifdef _DIR_PRIVATE_
	_DIR_PRIVATE_
#endif
} DIR;

struct dirent
{
	long		d_fileno;	/* entry serial number		*/
	int		d_reclen;	/* entry length			*/
	int		d_namlen;	/* entry name length		*/
	char		d_name[1];	/* entry name			*/
};

#ifndef _BLD_3d

#ifdef	rewinddir
#undef	rewinddir
#define rewinddir(p)	seekdir(p,0L)
#endif

extern DIR*		opendir(const char*);
extern void		closedir(DIR*);
extern struct dirent*	readdir(DIR*);
extern void		seekdir(DIR*, long);
extern long		telldir(DIR*);

#endif

#endif
