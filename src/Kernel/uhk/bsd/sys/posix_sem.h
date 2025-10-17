/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 22, 2025.
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
 * posix_shm.c : Support for POSIX semaphore APIs
 *
 *	File:	posix_sem.c
 *	Author:	Ananthakrishna Ramesh
 *
 * HISTORY
 * 2-Sep-1999	A.Ramesh
 *	Created for MacOSX
 *
 */

#ifndef _SYS_POSIX_SEM_H_
#define _SYS_POSIX_SEM_H_

#include <sys/appleapiopts.h>
#include <sys/types.h>
#include <sys/proc.h>

struct label;

#define PSEMNAMLEN      31      /* maximum name segment length we bother with */

struct pseminfo {
	unsigned int    psem_flags;
	unsigned int    psem_usecount;
	mode_t          psem_mode;
	uid_t           psem_uid;
	gid_t           psem_gid;
	char            psem_name[PSEMNAMLEN + 1];      /* segment name */
	void *          psem_semobject;
	struct label *  psem_label;
	pid_t           psem_creator_pid;
	uint64_t        psem_creator_uniqueid;
};

#define PSEMINFO_NULL (struct pseminfo *)0

#define PSEM_NONE       1
#define PSEM_DEFINED    2
#define PSEM_ALLOCATED  4
#define PSEM_MAPPED     8
#define PSEM_INUSE      0x10
#define PSEM_REMOVED    0x20
#define PSEM_INCREATE   0x40
#define PSEM_INDELETE   0x80

#endif
