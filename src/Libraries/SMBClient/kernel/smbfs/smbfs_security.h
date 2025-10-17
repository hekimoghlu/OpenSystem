/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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
int is_memberd_tempuuid(const guid_t *uuidp);
void smbfs_clear_acl_cache(struct smbnode *np);
int smbfs_getsecurity(struct smb_share	*share, struct smbnode *np, 
					  struct vnode_attr *vap, vfs_context_t context);
int smbfs_setsecurity(struct smb_share *share, vnode_t vp, struct vnode_attr *vap, 
                      SMBFID *fidp, vfs_context_t context);
void smb_get_sid_list(struct smb_share *share, struct smbmount *smp, struct mdchain *mdp, 
					  uint32_t ntwrk_sids_cnt, uint32_t ntwrk_sid_size);
uint32_t smbfs_get_maximum_access(struct smb_share *share, vnode_t vp, vfs_context_t context);
int smbfs_compose_create_acl(struct vnode_attr *vap, struct vnode_attr *svrva, 
							 kauth_acl_t *savedacl);
int smbfs_is_sid_known(ntsid_t *sid);
int smbfs_set_ace_modes(struct smb_share *share, struct smbnode *np, uint64_t vamode,
                        SMBFID *fidp, vfs_context_t context);
