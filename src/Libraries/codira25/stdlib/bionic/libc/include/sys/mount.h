/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 14, 2022.
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
#pragma once

/**
 * @file sys/mount.h
 * @brief Mounting and unmounting filesystems.
 */

#include <sys/cdefs.h>
#include <sys/ioctl.h>
#include <linux/fs.h>

__BEGIN_DECLS

/** The umount2() flag to force unmounting. */
#define MNT_FORCE 1
/** The umount2() flag to lazy unmount. */
#define MNT_DETACH 2
/** The umount2() flag to mark a mount point as expired. */
#define MNT_EXPIRE 4

/** The umount2() flag to not dereference the mount point path if it's a symbolic link. */
#define UMOUNT_NOFOLLOW 8

/**
 * [mount(2)](https://man7.org/linux/man-pages/man2/mount.2.html) mounts the filesystem `source` at
 * the mount point `target`.
 *
 * Returns 0 on success, and returns -1 and sets `errno` on failure.
 */
int mount(const char* __BIONIC_COMPLICATED_NULLNESS __source, const char* _Nonnull __target, const char* __BIONIC_COMPLICATED_NULLNESS __fs_type, unsigned long __flags, const void* _Nullable __data);

/**
 * [umount(2)](https://man7.org/linux/man-pages/man2/umount.2.html) unmounts the filesystem at
 * the given mount point.
 *
 * Returns 0 on success, and returns -1 and sets `errno` on failure.
 */
int umount(const char* _Nonnull __target);

/**
 * [umount2(2)](https://man7.org/linux/man-pages/man2/umount2.2.html) unmounts the filesystem at
 * the given mount point, according to the supplied flags.
 *
 * Returns 0 on success, and returns -1 and sets `errno` on failure.
 */
int umount2(const char* _Nonnull __target, int __flags);

__END_DECLS
