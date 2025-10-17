/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 29, 2025.
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
 *	Copyright (c) 1990, 1996-1998 Apple Computer, Inc.
 *	All Rights Reserved.
 */
/*
 * posix_shm.c : Support for POSIX shared memory APIs
 *
 *	File:	posix_shm.c
 *	Author:	Ananthakrishna Ramesh
 *
 * HISTORY
 * 2-Sep-1999	A.Ramesh
 *	Created for MacOSX
 *
 */

#ifndef _SYS_POSIX_SHM_H_
#define _SYS_POSIX_SHM_H_

#include <sys/appleapiopts.h>
#include <sys/types.h>
#include <sys/proc.h>

struct label;

#define PSHMNAMLEN      31      /* maximum name segment length we bother with */

struct pshminfo {
	unsigned int pshm_flags;
	unsigned int pshm_usecount;
	off_t        pshm_length;
	mode_t       pshm_mode;
	uid_t        pshm_uid;
	gid_t        pshm_gid;
	char         pshm_name[PSHMNAMLEN + 1];
	void         *pshm_memobject;
	struct label *pshm_label;
};

#endif
