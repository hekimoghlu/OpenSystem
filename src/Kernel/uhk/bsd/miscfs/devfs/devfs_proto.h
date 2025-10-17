/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 16, 2022.
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
/* THIS FILE HAS BEEN PRODUCED AUTOMATICALLY */
#ifndef __DEVFS_DEVFS_PROTO_H__
#define __DEVFS_DEVFS_PROTO_H__

#include  <sys/appleapiopts.h>

__BEGIN_DECLS
#ifdef __APPLE_API_PRIVATE
int     devfs_sinit(void);
devdirent_t *   dev_findname(devnode_t * dir, const char *name);
int     dev_add_name(const char * name, devnode_t * dirnode, devdirent_t * back,
    devnode_t * dnp, devdirent_t * *dirent_pp);
int     dev_add_node(int entrytype, devnode_type_t * typeinfo, devnode_t * proto,
    devnode_t * *dn_pp, struct devfsmount *dvm);
void    devnode_free(devnode_t * dnp);
int     dev_dup_plane(struct devfsmount *devfs_mp_p);
void    devfs_free_plane(struct devfsmount *devfs_mp_p);
int     dev_free_name(devdirent_t * dirent_p);
int     devfs_dntovn(devnode_t * dnp, struct vnode **vn_pp, struct proc * p);
int     dev_add_entry(const char *name, devnode_t * parent, int type, devnode_type_t * typeinfo,
    devnode_t * proto, struct devfsmount *dvm, devdirent_t * *nm_pp);
int     devfs_mount(struct mount *mp, vnode_t devvp, user_addr_t data,
    vfs_context_t context);
int     devfs_kernel_mount(char * mntname);
#endif /* __APPLE_API_PRIVATE */
__END_DECLS

#endif /* __DEVFS_DEVFS_PROTO_H__ */
/* THIS FILE PRODUCED AUTOMATICALLY */
/* DO NOT EDIT (see reproto.sh) */
