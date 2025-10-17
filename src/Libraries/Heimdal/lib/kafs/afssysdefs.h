/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 25, 2022.
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
/* $Id$ */

/*
 * This section is for machines using single entry point AFS syscalls!
 * and/or
 * This section is for machines using multiple entry point AFS syscalls!
 *
 * SunOS 4 is an example of single entry point and sgi of multiple
 * entry point syscalls.
 */

#if SunOS == 40
#define AFS_SYSCALL	31
#endif

#if SunOS >= 50 && SunOS < 57
#define AFS_SYSCALL	105
#endif

#if SunOS == 57
#define AFS_SYSCALL	73
#endif

#if SunOS >= 58
#define AFS_SYSCALL	65
#endif

#if defined(__hpux)
#define AFS_SYSCALL	50
#define AFS_SYSCALL2	49
#define AFS_SYSCALL3	48
#endif

#if defined(_AIX)
/* _AIX is too weird */
#endif

#if defined(__sgi)
#define AFS_PIOCTL      (64+1000)
#define AFS_SETPAG      (65+1000)
#endif

#if defined(__osf__)
#define AFS_SYSCALL	232
#define AFS_SYSCALL2	258
#endif

#if defined(__ultrix)
#define AFS_SYSCALL	31
#endif

#if defined(__FreeBSD__)
#if __FreeBSD_version >= 500000
#define AFS_SYSCALL 339
#else
#define AFS_SYSCALL 210
#endif
#endif /* __FreeBSD__ */

#ifdef __DragonFly__
#ifndef AFS_SYSCALL
#define AFS_SYSCALL 339
#endif
#endif

#ifdef __OpenBSD__
#define AFS_SYSCALL 208
#endif

#if defined(__NetBSD__)
#define AFS_SYSCALL 210
#endif

#ifdef __APPLE__		/* MacOS X */
#define AFS_SYSCALL 230
#endif

#ifdef SYS_afs_syscall
#define AFS_SYSCALL3	SYS_afs_syscall
#endif
