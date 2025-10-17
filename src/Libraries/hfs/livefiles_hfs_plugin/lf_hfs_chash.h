/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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
#ifndef lf_hfs_chash_h
#define lf_hfs_chash_h
#include "lf_hfs_common.h"
#include "lf_hfs.h"

struct cnode* hfs_chash_getcnode(struct hfsmount *hfsmp, ino_t inum, struct vnode **vpp, int wantrsrc, int skiplock, int *out_flags, int *hflags);
void hfs_chash_lock(struct hfsmount *hfsmp);
void hfs_chash_lock_spin(struct hfsmount *hfsmp);
void hfs_chash_lock_convert(struct hfsmount *hfsmp);
void hfs_chash_unlock(struct hfsmount *hfsmp);
void hfs_chashwakeup(struct hfsmount *hfsmp, struct cnode *cp, int hflags);
void hfs_chash_abort(struct hfsmount *hfsmp, struct cnode *cp);
struct vnode* hfs_chash_getvnode(struct hfsmount *hfsmp, ino_t inum, int wantrsrc, int skiplock, int allow_deleted);
int hfs_chash_snoop(struct hfsmount *hfsmp, ino_t inum, int existence_only, int (*callout)(const cnode_t *cp, void *), void * arg);
int hfs_chash_set_childlinkbit(struct hfsmount *hfsmp, cnid_t cnid);
int hfs_chashremove(struct hfsmount *hfsmp, struct cnode *cp);
void hfs_chash_mark_in_transit(struct hfsmount *hfsmp, struct cnode *cp);
void hfs_chash_lower_OpenLookupCounter(struct cnode *cp);
void hfs_chash_raise_OpenLookupCounter(struct cnode *cp);

#endif /* lf_hfs_chash_h */
