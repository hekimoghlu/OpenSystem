/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 16, 2022.
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
 * Copyright 2004 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#ifndef _SYS_PROCFS_H
#define	_SYS_PROCFS_H

/*
 * Possible values of pr_dmodel.
 * This isn't isa-specific, but it needs to be defined here for other reasons.
 */
#define PR_MODEL_UNKNOWN 0
#define PR_MODEL_ILP32  1       /* process data model is ILP32 */
#define PR_MODEL_LP64   2       /* process data model is LP64 */

#ifdef	__cplusplus
extern "C" {
#endif

/*
 * process status file.  /proc/<pid>/status
 */
typedef struct pstatus {
	int	pr_flags;	/* flags (see below) */
	pid_t	pr_pid;		/* process id */
	char	pr_dmodel;	/* data model of the process (see above) */
} pstatus_t;

/*
 * pr_flags (same values appear in both pstatus_t and lwpstatus_t pr_flags).
 *
 * These flags do *not* apply to psinfo_t.pr_flag or lwpsinfo_t.pr_flag
 * (which are both deprecated).
 */
/* The following flags apply to the specific or representative lwp */
#define PR_FORK    0x00100000   /* inherit-on-fork is in effect */
#define PR_RLC     0x00200000   /* run-on-last-close is in effect */
#define PR_KLC     0x00400000   /* kill-on-last-close is in effect */
#define PR_ASYNC   0x00800000   /* asynchronous-stop is in effect */
#define PR_BPTADJ  0x02000000   /* breakpoint trap pc adjustment is in effect */

typedef struct prmap {
	uint64_t pr_vaddr;	/* virtual address of mapping */
	int	pr_mflags;	/* protection and attribute flags (see below) */
} prmap_t;

/* Protection and attribute flags */
#define MA_READ         0x04    /* readable by the traced process */
#define MA_WRITE        0x02    /* writable by the traced process */
#define MA_EXEC         0x01    /* executable by the traced process */

#ifdef	__cplusplus
}
#endif

#endif	/* _SYS_PROCFS_H */
