/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 10, 2025.
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
#ifndef __COMPAT_SYS__TYPES_H_
#define __COMPAT_SYS__TYPES_H_

#include <stdint.h>

typedef int64_t       blkcnt_t;      /* total blocks */
typedef int32_t       blksize_t;     /* preferred block size */
typedef int32_t       dev_t;         /* dev_t */
typedef unsigned int  fsblkcnt_t;    /* Used by statvfs and fstatvfs */
typedef unsigned int  fsfilcnt_t;    /* Used by statvfs and fstatvfs */
typedef uint32_t      uid_t;         /* [???] user IDs */
typedef uint32_t      gid_t;         /* [???] process and group IDs */
typedef uint64_t      ino64_t;       /* [???] Used for 64 bit inodes */
typedef ino64_t       ino_t;         /* [???] Used for inodes */
typedef uint16_t      mode_t;        /* [???] Some file attributes */
typedef uint16_t      nlink_t;       /* link count */

#endif /* __COMPAT_SYS__TYPES_H_ */
