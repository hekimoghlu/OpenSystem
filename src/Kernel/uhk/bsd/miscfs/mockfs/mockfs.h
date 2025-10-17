/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 13, 2024.
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
#ifndef MOCKFS_H
#define MOCKFS_H

#if MOCKFS

#include <kern/locks.h>
#include <miscfs/mockfs/mockfs_fsnode.h>
#include <sys/kernel_types.h>

/*
 * mockfs is effectively a "fake" filesystem; the primary motivation for it being that we may have cases
 *   where our userspace needs are extremely simple/consistent and can provided by a single binary.  mockfs
 *   uses an in-memory tree to define the structure for an extremely simple filesystem, which makes the
 *   assumption that our root device is in fact a mach-o file, and provides a minimal filesystem to support
 *   this:  the root directory, a mountpoint for devfs (given that very basic userspace code may assume the
 *   existance of /dev/), and an executable representing the root device.
 *
 * The functionality supported by mockfs is minimal: it is read-only, and does not support user initiated IO,
 *   but it supports lookup (so it should be possible for the user to access /dev/).
 *
 * mockfs is primarily targeted towards memory-backed devices, and will (when possible) attempt to inform the
 *   VM that we are using a memory-backed device, so that we can eschew IO to the backing device completely,
 *   and avoid having an extra copy of the data in the UBC (as well as the overhead associated with creating
 *   that copy).
 *
 * For the moment, mockfs is not marked in vfs_conf.c as being threadsafe.
 */

struct mockfs_mount {
	lck_mtx_t       mockfs_mnt_mtx;         /* Mount-wide (and tree-wide) mutex */
	mockfs_fsnode_t mockfs_root;            /* Root of the node tree */
	boolean_t       mockfs_memory_backed;   /* Does the backing store reside in memory */
	boolean_t       mockfs_physical_memory; /* (valid if memory backed) */
	uint32_t        mockfs_memdev_base;     /* Base page of the backing store (valid if memory backed) */
	uint64_t        mockfs_memdev_size;     /* Size of the backing store (valid if memory backed) */
};

typedef struct mockfs_mount * mockfs_mount_t;

#endif /* MOCKFS */

#endif /* MOCKFS_H */
