/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 28, 2023.
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
#ifndef _SYS__TYPES_H_
#define _SYS__TYPES_H_

#include <sys/cdefs.h>
#include <machine/_types.h>

#if defined(KERNEL)
#ifdef XNU_KERNEL_PRIVATE
/*
 * Xcode doesn't currently set up search paths correctly for Kernel extensions,
 * so the clang headers are not seen in the correct order to use their types.
 */
#endif
#define USE_CLANG_TYPES 0
#else
#if defined(__has_feature) && __has_feature(modules)
#define USE_CLANG_TYPES 1
#else
#define USE_CLANG_TYPES 0
#endif
#endif

#if USE_CLANG_TYPES
#include <sys/_types/_null.h>
#endif

/*
 * Type definitions; takes common type definitions that must be used
 * in multiple header files due to [XSI], removes them from the system
 * space, and puts them in the implementation space.
 */

#if USE_CLANG_TYPES
#define __DARWIN_NULL NULL
#elif defined(__cplusplus)
#ifdef __GNUG__
#define __DARWIN_NULL __null
#else /* ! __GNUG__ */
#ifdef __LP64__
#define __DARWIN_NULL (0L)
#else /* !__LP64__ */
#define __DARWIN_NULL 0
#endif /* __LP64__ */
#endif /* __GNUG__ */
#else /* ! __cplusplus */
#define __DARWIN_NULL ((void *)0)
#endif

#if !defined(DRIVERKIT)
typedef __int64_t       __darwin_blkcnt_t;      /* total blocks */
typedef __int32_t       __darwin_blksize_t;     /* preferred block size */
typedef __int32_t       __darwin_dev_t;         /* dev_t */
typedef unsigned int    __darwin_fsblkcnt_t;    /* Used by statvfs and fstatvfs */
typedef unsigned int    __darwin_fsfilcnt_t;    /* Used by statvfs and fstatvfs */
typedef __uint32_t      __darwin_gid_t;         /* [???] process and group IDs */
typedef __uint32_t      __darwin_id_t;          /* [XSI] pid_t, uid_t, or gid_t*/
typedef __uint64_t      __darwin_ino64_t;       /* [???] Used for 64 bit inodes */
#if __DARWIN_64_BIT_INO_T
typedef __darwin_ino64_t __darwin_ino_t;        /* [???] Used for inodes */
#else /* !__DARWIN_64_BIT_INO_T */
typedef __uint32_t      __darwin_ino_t;         /* [???] Used for inodes */
#endif /* __DARWIN_64_BIT_INO_T */
typedef __darwin_natural_t __darwin_mach_port_name_t; /* Used by mach */
typedef __darwin_mach_port_name_t __darwin_mach_port_t; /* Used by mach */
typedef __uint16_t      __darwin_mode_t;        /* [???] Some file attributes */
typedef __int64_t       __darwin_off_t;         /* [???] Used for file sizes */
typedef __int32_t       __darwin_pid_t;         /* [???] process and group IDs */
typedef __uint32_t      __darwin_sigset_t;      /* [???] signal set */
typedef __int32_t       __darwin_suseconds_t;   /* [???] microseconds */
typedef __uint32_t      __darwin_uid_t;         /* [???] user IDs */
typedef __uint32_t      __darwin_useconds_t;    /* [???] microseconds */
#endif /* !defined(DRIVERKIT) */
typedef unsigned char   __darwin_uuid_t[16];
typedef char    __darwin_uuid_string_t[37];

#undef USE_CLANG_TYPES

#if !defined(KERNEL) && !defined(DRIVERKIT)
#include <sys/_pthread/_pthread_types.h>
#endif /* !defined(KERNEL) && !defined(DRIVERKIT) */

#if defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 5 || __GNUC__ > 3)
#define __offsetof(type, field) __builtin_offsetof(type, field)
#else /* !(gcc >= 3.5) */
#define __offsetof(type, field) ((size_t)(&((type *)0)->field))
#endif /* (gcc >= 3.5) */

#ifdef KERNEL
#include <sys/_types/_offsetof.h>
#endif /* KERNEL */

#endif  /* _SYS__TYPES_H_ */
