/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 15, 2024.
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
#ifndef lf_hfs_fsops_handler_h
#define lf_hfs_fsops_handler_h

#include "lf_hfs_common.h"
#include "lf_hfs_vnode.h"

#define PATH_TO_FSCK FS_BUNDLE_BIN_PATH "/fsck_hfs"

uint64_t FSOPS_GetOffsetFromClusterNum(vnode_t vp, uint64_t uClusterNum);
int      LFHFS_Mount   (int iFd, UVFSVolumeId puVolId, __unused UVFSMountFlags puMountFlags,
	__unused UVFSVolumeCredential *psVolumeCreds, UVFSFileNode *ppsRootNode);
int      LFHFS_Unmount (UVFSFileNode psRootNode, UVFSUnmountHint hint);
int      LFHFS_ScanVols (int iFd, UVFSScanVolsRequest *psRequest, UVFSScanVolsReply *psReply );
int      LFHFS_Taste ( int iFd );

extern UVFSFSOps HFS_fsOps;

#endif /* lf_hfs_fsops_handler_h */
