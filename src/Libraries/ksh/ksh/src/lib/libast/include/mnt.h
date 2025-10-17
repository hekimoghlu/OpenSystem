/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 10, 2022.
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
 * mounted filesystem scan interface
 */

#ifndef _MNT_H
#define _MNT_H	1

#undef	MNT_REMOTE			/* aix clash			*/
#define MNT_REMOTE	(1<<0)		/* remote mount			*/

typedef struct
{
	char*	fs;			/* filesystem name		*/
	char*	dir;			/* mounted dir			*/
	char*	type;			/* filesystem type		*/
	char*	options;		/* options			*/
	int	freq;			/* backup frequency		*/
	int	npass;			/* number of parallel passes	*/
	int	flags;			/* MNT_* flags			*/
} Mnt_t;

#if _BLD_ast && defined(__EXPORT__)
#define extern		__EXPORT__
#endif

extern void*	mntopen(const char*, const char*);
extern Mnt_t*	mntread(void*);
extern int	mntwrite(void*, const Mnt_t*);
extern int	mntclose(void*);

#undef	extern

#endif
