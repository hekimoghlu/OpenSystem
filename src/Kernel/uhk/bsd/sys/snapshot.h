/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 6, 2022.
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
#ifndef _SYS_SNAPSHOT_H_
#define _SYS_SNAPSHOT_H_

#ifndef KERNEL

#include <sys/cdefs.h>
#include <machine/_types.h>
#include <sys/_types/_size_t.h>
#include <_types/_uint32_t.h>
#include <sys/attr.h>
#include <Availability.h>

__BEGIN_DECLS

int fs_snapshot_create(int, const char *, uint32_t) __OSX_AVAILABLE(10.12) __IOS_AVAILABLE(10.0) __TVOS_AVAILABLE(10.0) __WATCHOS_AVAILABLE(3.0);

int fs_snapshot_list(int, struct attrlist  *, void *, size_t, uint32_t) __OSX_AVAILABLE(10.12) __IOS_AVAILABLE(10.0) __TVOS_AVAILABLE(10.0) __WATCHOS_AVAILABLE(3.0);

int fs_snapshot_delete(int, const char *, uint32_t) __OSX_AVAILABLE(10.12) __IOS_AVAILABLE(10.0) __TVOS_AVAILABLE(10.0) __WATCHOS_AVAILABLE(3.0);

int fs_snapshot_rename(int, const char *, const char *, uint32_t) __OSX_AVAILABLE(10.12) __IOS_AVAILABLE(10.0) __TVOS_AVAILABLE(10.0) __WATCHOS_AVAILABLE(3.0);

#endif /* !KERNEL */

/* fs_snapshot_mount() supported flags */
#define SNAPSHOT_MNT_NOSUID             0x00000008    /* same as MNT_NOSUID */
#define SNAPSHOT_MNT_NODEV              0x00000010    /* same as MNT_NODEV */
#define SNAPSHOT_MNT_DONTBROWSE         0x00100000    /* same as MNT_DONTBROWSE */
#define SNAPSHOT_MNT_IGNORE_OWNERSHIP   0x00200000    /* same as MNT_IGNORE_OWNERSHIP */
#define SNAPSHOT_MNT_NOFOLLOW           0x08000000    /* same as MNT_NOFOLLOW */

#ifndef KERNEL

int fs_snapshot_mount(int, const char *, const char *, uint32_t) __OSX_AVAILABLE(10.12) __IOS_AVAILABLE(10.0)       __TVOS_AVAILABLE(10.0) __WATCHOS_AVAILABLE(3.0);

int fs_snapshot_revert(int, const char *, uint32_t) __OSX_AVAILABLE(10.12) __IOS_AVAILABLE(10.0) __TVOS_AVAILABLE(10.0) __WATCHOS_AVAILABLE(3.0);

#ifdef PRIVATE
int fs_snapshot_root(int, const char *, uint32_t) __OSX_AVAILABLE(10.12.4) __IOS_AVAILABLE(10.3) __TVOS_AVAILABLE(10.3) __WATCHOS_AVAILABLE(3.3);
#endif

__END_DECLS

#endif /* !KERNEL */

#ifdef PRIVATE

#define SNAPSHOT_OP_CREATE 0x01
#define SNAPSHOT_OP_DELETE 0x02
#define SNAPSHOT_OP_RENAME 0x03
#define SNAPSHOT_OP_MOUNT  0x04
#define SNAPSHOT_OP_REVERT 0x05
#define SNAPSHOT_OP_ROOT   0x06

#endif

#endif /* !_SYS_SNAPSHOT_H_ */
