/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 26, 2022.
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
#ifndef MOCKFS_FSNODE_H
#define MOCKFS_FSNODE_H

#if MOCKFS

#include <sys/kernel_types.h>

/*
 * Types for the filesystem nodes; for the moment, these will effectively serve as unique
 * identifiers. This can be generalized later, but at least for the moment, the read-only
 * nature of the filesystem and the terse semantics (you have VDIR and VREG, and VREG
 * always represents the entire backing device) makes this sufficient for now.
 *
 * TODO: Should this include MOCKFS_SBIN?  Right now we tell lookup that when looking in
 *   MOCKFS_ROOT, "sbin" resolves back onto MOCKFS_ROOT; this is a handy hack for aliasing,
 *   but may not mesh well with VFS.
 */
enum mockfs_fsnode_type {
	MOCKFS_ROOT,
	MOCKFS_DEV,
	MOCKFS_FILE
};

/*
 * For the moment, pretend everything is a directory with support for two entries; the
 *   executable binary is a one-to-one mapping with the backing devnode, so this may
 *   actually be all we're interested in.
 *
 * Stash the filesize in here too (this is easier then looking at the devnode for every
 *   VREG access).
 */
struct mockfs_fsnode {
	uint64_t               size;    /* Bytes of data; 0 unless type is MOCKFS_FILE */
	uint8_t                type;    /* Serves as a unique identifier for now */
	mount_t                mnt;     /* The mount that this node belongs to */
	vnode_t                vp;      /* vnode for this node (if one exists) */
	struct mockfs_fsnode * parent;  /* Parent of this node (NULL for root) */
	                                /* TODO: Replace child_a/child_b with something more flexible */
	struct mockfs_fsnode * child_a; /* TEMPORARY */
	struct mockfs_fsnode * child_b; /* TEMPORARY */
};

typedef struct mockfs_fsnode * mockfs_fsnode_t;

/*
 * See mockfs_fsnode.c for function details.
 */
int mockfs_fsnode_create(mount_t mp, uint8_t type, mockfs_fsnode_t * fsnpp);
int mockfs_fsnode_destroy(mockfs_fsnode_t fsnp);
int mockfs_fsnode_adopt(mockfs_fsnode_t parent, mockfs_fsnode_t child);
int mockfs_fsnode_orphan(mockfs_fsnode_t fsnp);
int mockfs_fsnode_child_by_type(mockfs_fsnode_t parent, uint8_t type, mockfs_fsnode_t * child);
int mockfs_fsnode_vnode(mockfs_fsnode_t fsnp, vnode_t * vpp);
int mockfs_fsnode_drop_vnode(mockfs_fsnode_t fsnp);

#endif /* MOCKFS */

#endif /* MOCKFS_FSNODE_H */
