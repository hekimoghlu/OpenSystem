/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 10, 2025.
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
#ifndef lf_hfs_vnops_h
#define lf_hfs_vnops_h

#include "lf_hfs_cnode.h"
#include "lf_hfs.h"
#include "lf_hfs_generic_buf.h"
#include <UserFS/UserVFS.h>

/* VNOP_READDIR flags: */
#define VNODE_READDIR_EXTENDED    0x0001   /* use extended directory entries */
#define VNODE_READDIR_REQSEEKOFF  0x0002   /* requires seek offset (cookies) */
#define VNODE_READDIR_SEEKOFF32   0x0004   /* seek offset values should fit in 32 bits */
#define VNODE_READDIR_NAMEMAX     0x0008   /* For extended readdir, try to limit names to NAME_MAX bytes */

/*
 * flags for VNOP_BLOCKMAP
 */
#define VNODE_READ    0x01
#define VNODE_WRITE    0x02
#define VNODE_BLOCKMAP_NO_TRACK 0x04

void replace_desc(struct cnode *cp, struct cat_desc *cdp);
int  hfs_vnop_readdir(vnode_t vp, int *eofflag, int *numdirent, ReadDirBuff_s* psReadDirBuffer, uint64_t puCookie, int flags);
int  hfs_vnop_readdirattr(vnode_t vp, int *eofflag, int *numdirent, ReadDirBuff_s* psReadDirBuffer, uint64_t puCookie);
int  hfs_fsync(struct vnode *vp, int waitfor, hfs_fsync_mode_t fsyncmode);
int  hfs_vnop_remove(struct vnode* psParentDir,struct vnode *psFileToRemove, struct componentname* psCN, int iFlags);
int  hfs_vnop_rmdir(struct vnode *dvp, struct vnode *vp, struct componentname* psCN);
int  hfs_removedir(struct vnode *dvp, struct vnode *vp, struct componentname *cnp, int skip_reserve, int only_unlink);
int hfs_vnop_setattr(vnode_t vp, const UVFSFileAttributes *attr);
int hfs_update(struct vnode *vp, int options);
const struct cat_fork * hfs_prepare_fork_for_update(filefork_t *ff, const struct cat_fork *cf, struct cat_fork *cf_buf, uint32_t block_size);
int hfs_vnop_readlink(struct vnode *vp, void* data, size_t dataSize, size_t *actuallyRead);
int hfs_vnop_create(vnode_t a_dvp, vnode_t *a_vpp, struct componentname *a_cnp, UVFSFileAttributes* a_vap);
int hfs_vnop_mkdir(vnode_t a_dvp, vnode_t *a_vpp, struct componentname *a_cnp, UVFSFileAttributes* a_vap);
int hfs_makenode(struct vnode *dvp, struct vnode **vpp, struct componentname *cnp, UVFSFileAttributes *psGivenAttr);
int hfs_vnop_symlink(struct vnode *dvp, struct vnode **vpp, struct componentname *cnp, char* symlink_content, UVFSFileAttributes *attrp);

int hfs_removedir(struct vnode *dvp, struct vnode *vp, struct componentname *cnp, int skip_reserve, int only_unlink);
int hfs_removefile(struct vnode *dvp, struct vnode *vp, struct componentname *cnp, int flags, int skip_reserve, int allow_dirs, int only_unlink);
int hfs_vnop_renamex(struct vnode *fdvp,struct vnode *fvp, struct componentname *fcnp, struct vnode *tdvp, struct vnode *tvp, struct componentname *tcnp);
int hfs_vnop_link(vnode_t vp, vnode_t tdvp, struct componentname *cnp);
int hfs_removefile_callback(GenericLFBuf *psBuff, void *pvArgs);

int  hfs_vgetrsrc( struct vnode *vp, struct vnode **rvpp);
#endif /* lf_hfs_vnops_h */
