/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 30, 2023.
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
#ifndef lf_hfs_link_h
#define lf_hfs_link_h

#include "lf_hfs_catalog.h"
#include "lf_hfs_cnode.h"
#include "lf_hfs.h"

void    hfs_relorigin(struct cnode *cp, cnid_t parentcnid);
void    hfs_savelinkorigin(cnode_t *cp, cnid_t parentcnid);
void    hfs_privatedir_init(struct hfsmount * hfsmp, enum privdirtype type);
int     hfs_lookup_lastlink (struct hfsmount *hfsmp, cnid_t linkfileid, cnid_t *lastid, struct cat_desc *cdesc);
int     hfs_unlink(struct hfsmount *hfsmp, struct vnode *dvp, struct vnode *vp, struct componentname *cnp, int skip_reserve);
void    hfs_relorigins(struct cnode *cp);
int     hfs_makelink(struct hfsmount *hfsmp, struct vnode *src_vp, struct cnode *cp,struct cnode *dcp, struct componentname *cnp);
cnid_t  hfs_currentparent(cnode_t *cp, bool have_lock);
#endif /* lf_hfs_link_h */
